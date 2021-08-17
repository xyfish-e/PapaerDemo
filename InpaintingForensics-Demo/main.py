import os
import cv2
import random
import argparse
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

# 可选择搜索模块
operation_canditates = {
    '00': lambda filter1, filter2, stride, dilation: SeparableConv2d(filter1, filter2, 3, stride, dilation),
    '01': lambda filter1, filter2, stride, dilation: SepConv(filter1, filter2, 3, stride, 1),
    '02': lambda filter1, filter2, stride, dilation: SepConv(filter1, filter2, 5, stride, 2),
    '03': lambda filter1, filter2, stride, dilation: Identity(),
}

# pf层权重
def get_pf_list():
    pf1 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0]]).astype('float32')

    pf2 = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]]).astype('float32')

    pf3 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]]).astype('float32')

    return [torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone()
            ]

# 约束权重
def constrained_weights(weights):
    weights = weights.permute(2, 3, 0, 1)  # 维度换位
    # Scale by 10k to avoid numerical issues while normalizing（避免归一化出现数值问题）
    weights = weights * 10000

    # Set central values to zero to exlude them from the normalization step
    weights[2, 2, :, :] = 0  # 所有通道卷积核中心值置为0

    # Pass the weights
    filter_1 = weights[:, :, 0, 0]  # 通道1
    filter_2 = weights[:, :, 0, 1]  # 通道2
    filter_3 = weights[:, :, 0, 2]  # 通道3

    # Normalize the weights for each filter.
    # Sum in the 3rd dimension, which contains 25 numbers.
    filter_1 = filter_1.reshape(1, 1, 1, 25)
    filter_1 = filter_1 / filter_1.sum(3).reshape(1, 1, 1, 1)  # 归一化
    filter_1[0, 0, 0, 12] = -1  # 中间通道置为0

    filter_2 = filter_2.reshape(1, 1, 1, 25)
    filter_2 = filter_2 / filter_2.sum(3).reshape(1, 1, 1, 1)
    filter_2[0, 0, 0, 12] = -1

    filter_3 = filter_3.reshape(1, 1, 1, 25)
    filter_3 = filter_3 / filter_3.sum(3).reshape(1, 1, 1, 1)
    filter_3[0, 0, 0, 12] = -1

    # Prints are for debug reasons.
    # The sums of all filter weights for a specific filter
    # should be very close to zero.
    # print(filter_1)
    # print(filter_2)
    # print(filter_3)
    # print(filter_1.sum(3).reshape(1,1,1,1))
    # print(filter_2.sum(3).reshape(1,1,1,1))
    # print(filter_3.sum(3).reshape(1,1,1,1))

    # Reshape to original size.
    filter_1 = filter_1.reshape(1, 1, 5, 5)  # 转换成5×5卷积核形状
    filter_2 = filter_2.reshape(1, 1, 5, 5)
    filter_3 = filter_3.reshape(1, 1, 5, 5)

    # Pass the weights back to the original matrix and return.
    weights[:, :, 0, 0] = filter_1
    weights[:, :, 0, 1] = filter_2
    weights[:, :, 0, 2] = filter_3

    weights = weights.permute(2, 3, 0, 1)
    return weights


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

# 局部相似性
class CustomizedConv(nn.Module):
    def __init__(self, channels=1, choice='similarity'):
        super(CustomizedConv, self).__init__()
        self.channels = channels
        self.choice = choice
        # 局部注意力的权重矩阵
        kernel = [[0.03598, 0.03735, 0.03997, 0.03713, 0.03579],
                  [0.03682, 0.03954, 0.04446, 0.03933, 0.03673],
                  [0.03864, 0.04242, 0.07146, 0.04239, 0.03859],
                  [0.03679, 0.03936, 0.04443, 0.03950, 0.03679],
                  [0.03590, 0.03720, 0.04003, 0.03738, 0.03601]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 增加两个通道
        kernel = np.repeat(kernel, self.channels, axis=0)  # 复制kernel至每个通道
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.kernel = nn.modules.utils._pair(3)  # (3, 3)
        self.stride = nn.modules.utils._pair(1)  # (1, 1)
        self.padding = nn.modules.utils._quadruple(0)  # (0, 0, 0, 0)
        self.same = False

    def __call__(self, x):
        if self.choice == 'median':
            x = F.pad(x, self._padding(x), mode='reflect')  # 对数据集图像或中间层特征进行维度扩充（mode=反射）
            x = x.unfold(2, self.kernel[0], self.stride[0]).unfold(3, self.kernel[1], self.stride[1])  # 手动实现的滑动窗口操作，也就是只有卷，没有积
            x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        else:
            x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=False),
        )

    def forward(self, x):
        return self.op(x)

# 恒等映射
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# 自定义数据集（继承Dataset）
# IID_Dataset(self.train_num, self.train_file, choice='train')
class IID_Dataset(Dataset):
    def __init__(self, num, file, choice='train'):
        self.num = num
        self.choice = choice
        # 读取排序后的文件列表
        if self.choice != 'test':
            try:
                self.filelist = np.load(file)
            except Exception:
                self.filelist = sorted(os.listdir('demo_input/'))
        else:
            self.filelist = sorted(os.listdir('demo_input/'))

        # 数据预处理（张量化、归一化）
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # 读取数据图像、掩码及其文件名
    def __getitem__(self, idx):
        return self.load_item(idx)

    # 获取数据集样本数
    def __len__(self):
        return self.num

    # 获取数据图像、掩码、文件名
    def load_item(self, idx):
        if self.choice != 'test':
            fname1, fname2 = self.filelist[idx]
        else:
            fname1, fname2 = 'demo_input/' + self.filelist[idx], ''

        img = cv2.imread(fname1)  # 获取图像文件
        H, W, _ = img.shape  # 获取高宽

        # 设置mask
        if fname2 == '':
            mask = np.zeros([H, W, 3])
            mask[np.random.randint(5, H-5), np.random.randint(5, W-5), 0] = 255
        else:
            mask = cv2.imread(fname2)

        # 对训练集进行随机翻转
        if self.choice == 'train':
            if random.random() < 0.5:
                img = cv2.flip(img, 0)  # 0垂直翻转
                mask = cv2.flip(mask, 0)
            if random.random() < 0.5:
                img = cv2.flip(img, 1)  # 1水平翻转
                mask = cv2.flip(mask, 1)

        # 归一化
        img = img.astype('float') / 255.
        mask = mask.astype('float') / 255.
        return self.transform(img), self.tensor(mask[:, :, :1]), fname1.split('/')[-1]

    # 图像维度 HWC → CHW
    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)

# 全局及局部注意力模块
class Global_Local_Attention(nn.Module):
    def __init__(self):
        super(Global_Local_Attention, self).__init__()
        self.local_att = CustomizedConv(256, choice='similarity')

    def forward(self, x):
        tmp = x
        former = tmp.permute(0, 2, 3, 1).view(tmp.size(0), tmp.size(2) * tmp.size(3), -1)
        num_0 = torch.einsum('bik,bjk->bij', [former, former])  # 爱因斯坦简记法，求和运算
        norm_former = torch.einsum("bij,bij->bi", [former, former])
        den_0 = torch.sqrt(torch.einsum('bi,bj->bij', [norm_former, norm_former]))
        cosine = num_0 / den_0

        F_local = self.local_att(x.clone())

        top_T = 15  # The default maximum value of T is 15
        cosine_max, indexes = torch.topk(cosine, top_T, dim=2)
        dy_T = top_T
        for t in range(top_T):
            if torch.mean(cosine_max[:, :, t]) >= 0.5:
                dy_T = t
        dy_T = max(2, dy_T)

        mask = torch.ones(tmp.size(0), tmp.size(2) * tmp.size(3))
        # mask = torch.ones(tmp.size(0), tmp.size(2) * tmp.size(3)).cuda()
        mask_index = (mask == 1).nonzero()[:, 1].view(tmp.size(0), -1)
        idx_b = torch.arange(tmp.size(0)).long().unsqueeze(1).expand(tmp.size(0), mask_index.size(1))

        rtn = tmp.clone().permute(0, 2, 3, 1).view(tmp.size(0), tmp.size(2) * tmp.size(3), -1)
        for t in range(1, dy_T):
            mask_index_top = (mask == 1).nonzero()[:, 1].view(tmp.size(0), -1).gather(1, indexes[:, :, t])
            ind_1st_top = torch.zeros(tmp.size(0), tmp.size(2) * tmp.size(3), tmp.size(2) * tmp.size(3))
            # ind_1st_top = torch.zeros(tmp.size(0), tmp.size(2) * tmp.size(3), tmp.size(2) * tmp.size(3)).cuda()
            ind_1st_top[(idx_b, mask_index, mask_index_top)] = 1
            rtn += torch.bmm(ind_1st_top, former)
        rtn = rtn / dy_T
        F_global = rtn.permute(0, 2, 1).view(tmp.shape)
        # The following line maybe useful when the location of Attention() in the network is changed.
        # F_global = nn.UpsamplingNearest2d(size=(x.shape[2], x.shape[3]))(rtn.float())
        x = torch.cat([x, F_global, F_local], dim=1)
        return x

# 可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

# 网络块
class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, genotype=None):
        super(Block, self).__init__()
        if not genotype:
            genotype = ['03', '03', '03']

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU()
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(operation_canditates[genotype[i]](filters, filters, 1, dilation))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        rep.append(self.relu)
        rep.append(operation_canditates[genotype[2]](filters, filters, stride, 1))
        rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x

# 网络
class IID_Net(nn.Module):
    def __init__(self):
        super(IID_Net, self).__init__()
        BatchNorm = nn.BatchNorm2d

        # The Enhancement Block
        self.normal_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.pf_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
        self.bayar_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1 = nn.Conv2d(15, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        # The Extraction Block
        # The code in 'genotype' means the best architecture selected from 1000 sampled candidate models.
        self.cell1 = Block(64, 128, reps=3, stride=2, BatchNorm=BatchNorm, start_with_relu=False, genotype=['01', '03', '00'])
        self.cell2 = Block(128, 256, reps=3, stride=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['02', '00', '00'])
        self.cell3 = Block(256, 256, reps=3, stride=1, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['03', '02', '00'])
        self.cell4 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['01', '00', '01'])
        self.cell5 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['00', '02', '00'])
        self.cell6 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['00', '01', '00'])
        self.cell7 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['02', '03', '02'])
        self.cell8 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['03', '03', '00'])
        self.cell9 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['02', '02', '00'])
        self.cell10 = Block(256, 256, reps=3, stride=1, dilation=2, BatchNorm=BatchNorm, start_with_relu=True, grow_first=True, genotype=['00', '01', '03'])

        # The Decision Block
        self.att = Global_Local_Attention()
        self.decision = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256 * 3, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 1, 3, stride=1, padding=1),
        )

        self.pf_list = get_pf_list()
        self.reset_pf()
        self.median = CustomizedConv(choice='median')

    def forward(self, x):
        _, _, H, W = x.shape

        # The Enhancement Block
        self.bayar_conv.weight.data = constrained_weights(self.bayar_conv.weight.data)
        bayar_x = self.bayar_conv(x)
        normal_x = self.normal_conv(x)
        pf_x = self.pf_conv(x)
        x = torch.cat([normal_x, bayar_x, pf_x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # The Extraction Block
        x = self.cell1(x)
        x = self.cell2(x)
        x = self.cell3(x)
        x = self.cell4(x)
        x = self.cell5(x)
        x = self.cell6(x)
        x = self.cell7(x)
        x = self.cell8(x)
        x = self.cell9(x)
        x = self.cell10(x)

        # The Decision Block
        x = self.att(x)
        x = self.decision(x)

        # The Median Filter is embedded here because we found that in some rare cases it would cause overflow
        # when calculating the loss functions, and this makes almost no difference.
        x = self.median(x)
        x = F.interpolate(x, (H, W)) # 上采样
        x = nn.Sigmoid()(x)
        return x

    def reset_pf(self):
        for idx, pf in enumerate(self.pf_list):
            self.pf_conv.weight.data[idx, :, :, :] = pf


# Used only for training one-shot NAS
class ChosenOperation_NAS(nn.Module):
    def __init__(self, C1, C2, stride, dilation):
        super(ChosenOperation_NAS, self).__init__()
        self._ops = nn.ModuleList()
        self.typelist = ['00', '01', '02', '03']
        for genotype in self.typelist:
            op = operation_canditates[genotype](C1, C2, stride, dilation)
            self._ops.append(op)

    def forward(self, x, genotype=None):
        weights = [1 for _ in range(len(self.typelist))]
        if genotype:
            weights = [0 for _ in range(len(self.typelist))]
            weights[self.typelist.index(genotype)] = 1
        return sum(w * op(x) for w, op in zip(weights, self._ops))


# Used only for training one-shot NAS
class Block_NAS(nn.Module):
    def __init__(self, inplanes, planes, reps=3, stride=1, dilation=2, BatchNorm=nn.BatchNorm2d):
        super(Block_NAS, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.reps = reps
        self.stride = stride
        self.dilation = dilation
        self.BatchNorm = BatchNorm

    def forward(self, x, genotype=None):
        x_skip = x
        if self.planes != self.inplanes or self.stride != 1:
            x_skip = nn.Conv2d(self.inplanes, self.planes, 1, stride=self.stride, bias=False)(x_skip)
            # x_skip = nn.Conv2d(self.inplanes, self.planes, 1, stride=self.stride, bias=False).cuda()(x_skip)
            x_skip = self.BatchNorm(self.planes)(x_skip)
            # x_skip = self.BatchNorm(self.planes).cuda()(x_skip)

        self.relu = nn.ReLU()
        x = self.relu(x)
        x = SeparableConv2d(self.inplanes, self.planes, stride=1, dilation=self.dilation)(x)
        # x = SeparableConv2d(self.inplanes, self.planes, stride=1, dilation=self.dilation).cuda()(x)
        x = self.BatchNorm(self.planes)(x)
        # x = self.BatchNorm(self.planes).cuda()(x)
        for i in range(self.reps - 1):
            x = self.relu(x)
            x = ChosenOperation_NAS(self.planes, self.planes, 1, self.dilation)(x, genotype[i])
            # x = ChosenOperation_NAS(self.planes, self.planes, 1, self.dilation).cuda()(x, genotype[i])
            x = self.BatchNorm(self.planes)(x)
            # x = self.BatchNorm(self.planes).cuda()(x)
        x = self.relu(x)
        x = ChosenOperation_NAS(self.planes, self.planes, self.stride, 1)(x, genotype[-1])
        # x = ChosenOperation_NAS(self.planes, self.planes, self.stride, 1).cuda()(x, genotype[-1])
        x = self.BatchNorm(self.planes)(x)
        # x = self.BatchNorm(self.planes).cuda()(x)
        x = x + x_skip
        return x


# Used only for training one-shot NAS
class IID_Net_NAS(nn.Module):
    def __init__(self):
        super(IID_Net_NAS, self).__init__()
        self.EnhancementBlock = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.ExtractBlock = nn.ModuleList()
        self.ExtractBlock += [Block_NAS(64, 128, stride=2)]
        self.ExtractBlock += [Block_NAS(128, 256, stride=2)]
        self.ExtractBlock += [Block_NAS(256, 256, stride=1)]
        self.ExtractBlock += [Block_NAS(256, 256, stride=1)]
        self.ExtractBlock += [Block_NAS(256, 256, stride=1)]
        self.ExtractBlock += [Block_NAS(256, 256, stride=1)]
        self.ExtractBlock += [Block_NAS(256, 256, stride=1)]
        self.ExtractBlock += [Block_NAS(256, 256, stride=1)]
        self.ExtractBlock += [Block_NAS(256, 256, stride=1)]
        self.ExtractBlock += [Block_NAS(256, 256, stride=1)]

        self.DecisionBlock = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, genotype=None):
        x = self.EnhancementBlock(x)
        for i, cell in enumerate(self.ExtractBlock):
            x = cell(x, genotype[i])
        x = self.DecisionBlock(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean', ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # self.crit = nn.BCELoss(reduction='none')
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

# 模型
class IID_Model(nn.Module):
    def __init__(self):
        super(IID_Model, self).__init__()
        self.lr = 1e-4  # 学习率
        self.networks = IID_Net()  # 网络
        # self.networks = IID_Net_NAS()
        pytorch_total_params = sum(p.numel() for p in self.networks.parameters() if p.requires_grad)
        print('Total Params: %d' % pytorch_total_params)
        with open('log.txt', 'a+') as f:
            f.write('\n\nIID-Net, Total Params: %d' % pytorch_total_params)
        self.gen = nn.DataParallel(self.networks)  # 分布式网络
        # self.gen = nn.DataParallel(self.networks).cuda()
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.save_dir = 'weights/'

    # 梯度清零，返回掩码及损失
    def process(self, Ii, Mg):
        self.gen_optimizer.zero_grad()  # 梯度清零

        Mo = self(Ii)  # 输出掩码

        gen_loss = FocalLoss()(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1).float())
        # gen_loss += nn.BCELoss()(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1))
        gen_loss += nn.BCEWithLogitsLoss()(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1))
        return Mo, gen_loss

    def forward(self, Ii):
        return self.gen(Ii)

    # 反向传播，更新梯度
    def backward(self, gen_loss=None):
        if gen_loss:
            gen_loss.backward(retain_graph=False)
            self.gen_optimizer.step()

    # 保存模型参数
    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + 'IID_weights.pth')

    # 读取保存的模型参数
    def load(self, path=''):
        self.gen.load_state_dict(torch.load(self.save_dir + path + 'IID_weights.pth', map_location='cpu'))
        # self.gen.load_state_dict(torch.load(self.save_dir + path + 'IID_weights.pth'))


class InpaintingForensics():
    def __init__(self):
        # self.train_num = 48000
        self.train_num = 800
        self.val_num = 200
        self.test_num = 12
        self.batch_size = 24
        # For training, please provide the absolute path of training data that saved in numpy with following format
        # E.g., file = [['./training_input_1.png', './training_ground_truth_1.png'],
        #              ['./training_input_2.png', './training_ground_truth_2.png'],...]
        # self.train_file = ''
        self.train_file = 'train_dataset.npy'
        self.val_file = 'val_dataset.npy'
        self.test_file = ''
        # 获取数据集
        train_dataset = IID_Dataset(self.train_num, self.train_file, choice='train')
        val_dataset = IID_Dataset(self.val_num, self.val_file, choice='val')
        test_dataset = IID_Dataset(self.test_num, self.test_file, choice='test')

        # self.giid_model = IID_Model().cuda()
        # 模型
        self.giid_model = IID_Model()
        # 训练周期
        self.n_epochs = 1000
        # 数据迭代器
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 训练
    def train(self):
        with open('log.txt', 'a+') as f:
            f.write('\nTrain %s with %d' % (self.train_file, self.train_num))
            f.write('\nVal %s with %d' % (self.val_file, self.val_num))
            f.write('\nTest %s with %d' % (self.test_file, self.test_num))

        # 学习率更新策略
        """
            1.monitor：被监测的量
            2.factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
            3.patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
            4.mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
            5.epsilon：阈值，用来确定是否进入检测值的“平原区”
            6.cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
            7.min_lr：学习率的下限
        """
        scheduler_gan = ReduceLROnPlateau(self.giid_model.gen_optimizer, patience=10, factor=0.5)

        best_auc = 0  # 最优准确率
        for epoch in range(self.n_epochs):
            cnt, gen_losses, auc = 0, [], []
            for items in self.train_loader:
                cnt += self.batch_size

                self.giid_model.train()
                Ii, Mg = (item for item in items[:-1])
                # Ii, Mg = (item.cuda() for item in items[:-1])
                Mo, gen_loss = self.giid_model.process(Ii, Mg)
                self.giid_model.backward(gen_loss)
                gen_losses.append(gen_loss.item())
                Mg, Mo = self.convert2(Mg), self.convert2(Mo)
                N, H, W, C = Mg.shape

                auc.append(roc_auc_score(Mg.reshape(N * H * W * C).astype('int'), Mo.reshape(N * H * W * C)) * 100.)
                print('Tra (%d/%d): G:%6.3f A:%3.2f' % (cnt, self.train_num, np.mean(gen_losses), np.mean(auc)), end='\r')
                if cnt % 12000 == 0 or cnt >= self.train_num:
                    val_gen_loss, val_auc = self.val()
                    scheduler_gan.step(val_auc)
                    print('Val (%d/%d): G:%6.3f A:%3.2f' % (cnt, self.train_num, val_gen_loss, val_auc))
                    if val_auc > best_auc:
                        best_auc = val_auc
                        self.giid_model.save('best/')
                    self.giid_model.save('latest/')
                    with open('log.txt', 'a+') as f:
                        f.write('\n(%d/%d): Tra: A:%4.2f Val: A:%4.2f' % (cnt, self.train_num, np.mean(auc), val_auc))
                    auc, gen_losses = [], []

    def val(self):
        self.giid_model.eval()
        auc, gen_losses = [], []
        for cnt, items in enumerate(self.val_loader):
            Ii, Mg = (item for item in items[:-1])
            # Ii, Mg = (item.cuda() for item in items[:-1])
            filename = items[-1][0]
            Mo, gen_loss = self.giid_model.process(Ii, Mg)
            gen_losses.append(gen_loss.item())
            Ii, Mg, Mo = self.convert1(Ii), self.convert2(Mg)[0], self.convert2(Mo)[0]
            H, W, _ = Mg.shape
            auc.append(roc_auc_score(Mg.reshape(H * W).astype('int'), Mo.reshape(H * W)) * 100.)

            # Sample 100 validation images for visualization
            if len(auc) <= 100:
                Mg, Mo = Mg * 255, Mo * 255
                out = np.zeros([H, H * 3, 3])
                out[:, :H, :] = Ii
                out[:, H:H*2, :] = np.concatenate([Mo, Mo, Mo], axis=2)
                out[:, H*2:, :] = np.concatenate([Mg, Mg, Mg], axis=2)
                cv2.imwrite('demo_val/val_' + filename, out)
        return np.mean(gen_losses), np.mean(auc)

    def test(self):
        self.giid_model.load()
        self.giid_model.eval()
        for cnt, items in enumerate(self.test_loader):
            print(cnt, end='\r')
            Ii, Mg = (item for item in items[:-1])
            # Ii, Mg = (item.cuda() for item in items[:-1])
            filename = items[-1][0]
            Mo, gen_loss = self.giid_model.process(Ii, Mg)
            Ii, Mo = self.convert1(Ii), self.convert2(Mo)[0]
            cv2.imwrite('demo_output/output_' + filename, Mo * 255)

    def convert1(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
        return img

    def convert2(self, x):
        return x.permute(0, 2, 3, 1).cpu().detach().numpy()


if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='train or test the model', choices=['train', 'test'])
    args = parser.parse_args()
    # 模型
    model = InpaintingForensics()
    if args.type == 'train':
        model.train()
    elif args.type == 'test':
        model.test()
