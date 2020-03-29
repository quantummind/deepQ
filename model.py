import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import itertools
import numpy as np

class CircuitPairDataset(Dataset):
    # shape of data: [[circuit 0 equiv 0 data, circuit 0 equiv 1 data, ...], ...]
        # where each internal data is length n_channels
    # shape of target: [[circuit 0 equiv 0 noise, circuit 0 equiv 1 noise, ...], ...]
    # reflection assumes circuit topology is symmetric about middle qubit
    def __init__(self, data, target, transform=None, reflect=False):
        self.augmentation_factor = 1
        if reflect:
            self.augmentation_factor *= 2
        self.reflect = reflect
        
        old_shape = data.shape[0:2]
        new_shape = (old_shape[0]*old_shape[1],)
        self.data = torch.from_numpy(np.reshape(data, (new_shape[0], data.shape[-3], data.shape[-2], data.shape[-1]))).float()
        self.target = torch.from_numpy(target.flatten()).float()
        self.transform = transform
        
        self.circuit_pairs = []
        for i in range(len(data)):
            row = data[i]
            row_inds = np.zeros(len(row)).astype(int) + i
            col_inds = np.arange(len(row))
            flat_inds = np.ravel_multi_index((row_inds, col_inds), old_shape)
            pairs = list(itertools.product(flat_inds, flat_inds))
            for j in reversed(range(len(flat_inds))):
                del pairs[(len(flat_inds)+1) * j]
            self.circuit_pairs.extend(pairs)
    
    def __getitem__(self, index):
        reflected = False
        if self.reflect and index >= len(self.circuit_pairs):
            reflected = True
            index -= len(self.circuit_pairs)
        
        i1, i2 = self.circuit_pairs[index]
        x1 = self.data[i1]
        y1 = self.target[i1]
        x2 = self.data[i2]
        y2 = self.target[i2]
        
        # only get change in CNOT count
        n2 = torch.sum(x2[:, -1])
        n1 = torch.sum(x1[:, -1])
        delta_gate_count = (n2 - n1).view(-1)
        
        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
            
        if reflected:
            x1 = torch.flip(x1, [2])
            x2 = torch.flip(x2, [2])
        
        x = torch.cat((x1, x2), 0)
        y = (y2 - y1).view(-1)
        return x, delta_gate_count, y
    
    def __len__(self):
        return len(self.circuit_pairs)*self.augmentation_factor
    

    
class SmallNet(nn.Module):
    def __init__(self, n_channels=4, concat_features=False):
        super(SmallNet, self).__init__()
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(n_channels, n_channels, 5, padding=(0, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=(0, 1))
        if concat_features:
            self.fc1 = nn.Linear(n_channels*2 * 2 + concat_data_len, n_channels*4)
        else:
            self.fc1 = nn.Linear(72, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x, concat_data=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 72)
        if concat_data is not None:
            x = torch.cat((x, concat_data), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x