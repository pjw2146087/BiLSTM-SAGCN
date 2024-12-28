import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
import os
import time

class TraceDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train', shuffle=False, seed_value=None):
        assert mode in ['train', 'eval', 'test'], 'mode is one of train, eval.'
        self.mode = mode
        self.data = []
        self.adjs = {}

        if mode == 'train' or mode == 'eval':
            input_folder = path
            excel_files = glob.glob(os.path.join(input_folder, "*.xlsx"))
            i = 0

            for filename in excel_files:
                i += 1
                data = pd.read_excel(filename, header=0)
                data = data.assign(new_column=i)
                points = np.array(data)[:,:-2].astype('float32')
                labels = np.array(data)[:, -2].reshape(-1, 1).astype('int64')
                trace_id = np.array(data)[:, -1].astype('int64')
                self.data.append((points, labels, trace_id))
                self.adjs[str(i)] = np.load(filename[0:-4]+'npy')

            if shuffle:
                if seed_value is None:
                    seed_value = int(time.time())
                np.random.seed(int(seed_value))
                np.random.shuffle(self.data)

        else:
            input_folder = path
            excel_files = glob.glob(os.path.join(input_folder, "*.xlsx"))
            i = 0

            for filename in excel_files:
                i += 1
                data = pd.read_excel(filename, header=0)
                data = data.assign(new_column=i)
                points = np.array(data)[:,:-1].astype('float32')
                trace_id = np.array(data)[:, -1].astype('int64')
                self.data.append((points, trace_id))
                self.adjs[str(i)] = np.load(filename[0:-4]+'npy')

            if shuffle:
                if seed_value is None:
                    seed_value = int(time.time())
                np.random.seed(int(seed_value))
                np.random.shuffle(self.data)

    def __getitem__(self, index):
        if self.mode in ['train', 'eval']:
            points, labels, trace_id = self.data[index]
            return points, labels, trace_id
        else:
            points, trace_id = self.data[index]
            return points, trace_id

    def __getadj__(self, traid):
        return self.adjs[str(traid)]

    def __len__(self):
        return len(self.data)
