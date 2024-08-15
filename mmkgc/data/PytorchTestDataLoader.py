import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TestDataSampler(object):

    def __init__(self, data_total, data_sampler):
        self.data_total = data_total
        self.data_sampler = data_sampler
        self.total = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.total += 1
        if self.total > self.data_total:
            raise StopIteration()
        return self.data_sampler()

    def __len__(self):
        return self.data_total

class TestDataset(Dataset):

    def __init__(self, in_path="./", sampling_mode='link', type_constrain=True):
        self.in_path = in_path
        self.sampling_mode = sampling_mode
        self.type_constrain = type_constrain
        self.read()

    def read(self):
        if self.in_path != "./":
            self.tri_file = os.path.join(self.in_path, "test2id.txt")
            self.ent_file = os.path.join(self.in_path, "entity2id.txt")
            self.rel_file = os.path.join(self.in_path, "relation2id.txt")

            self.relTotal = len(open(self.rel_file).readlines())
            self.entTotal = len(open(self.ent_file).readlines())
            self.testTotal = len(open(self.tri_file).readlines())

    def sampling_lp(self):
        # Implement your logic for link prediction sampling here
        pass

    def sampling_tc(self):
        # Implement your logic for triple classification sampling here
        pass

    def __len__(self):
        return self.testTotal

    def __getitem__(self, idx):
        if self.sampling_mode == "link":
            # Implement your logic for link prediction sampling here
            pass
        else:
            # Implement your logic for triple classification sampling here
            pass
