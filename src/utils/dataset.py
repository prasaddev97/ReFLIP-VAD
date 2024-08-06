import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, ReFLIP_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.ReFLIP_dim = ReFLIP_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        ReFLIP_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            ReFLIP_feature, ReFLIP_length = tools.process_feat(ReFLIP_feature, self.ReFLIP_dim)
        else:
            ReFLIP_feature, ReFLIP_length = tools.process_split(ReFLIP_feature, self.ReFLIP_dim)

        ReFLIP_feature = torch.tensor(ReFLIP_feature)
        ReFLIP_label = self.df.loc[index]['label']
        return ReFLIP_feature, ReFLIP_label, ReFLIP_length

class XDDataset(data.Dataset):
    def __init__(self, ReFLIP_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.ReFLIP_dim = ReFLIP_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        ReFLIP_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            ReFLIP_feature, ReFLIP_length = tools.process_feat(ReFLIP_feature, self.ReFLIP_dim)
        else:
            ReFLIP_feature, ReFLIP_length = tools.process_split(ReFLIP_feature, self.ReFLIP_dim)

        ReFLIP_feature = torch.tensor(ReFLIP_feature)
        ReFLIP_label = self.df.loc[index]['label']
        return ReFLIP_feature, ReFLIP_label, ReFLIP_length