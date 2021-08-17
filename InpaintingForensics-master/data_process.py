import os
import numpy as np

def process_file(file_path='E:/python_projectDemo/Paper/InpaintingForensics-master/DiverseInpaintingDataset/', file_type='CA/', split_rate=0.2):
    path = os.path.join(file_path, file_type)
    pathFile = os.listdir(path)
    traindata = [[os.path.join(path, pathFile[i]), os.path.join(path, pathFile[i+1])] for i in range(0, int(len(pathFile)*(1-split_rate)), 2)]
    valdata = [[os.path.join(path, pathFile[i]), os.path.join(path, pathFile[i+1])] for i in range(int(len(pathFile)*(1-split_rate)), len(pathFile), 2)]
    np.save('train_dataset.npy', traindata)
    np.save('val_dataset.npy', valdata)
    return