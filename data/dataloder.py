import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class load_dataset(Dataset):
    def __init__(self, dataset, configs):
        super(load_dataset, self).__init__()
        self.configs = configs
        data = dataset['data']
        self.Z = dataset['Z']
        self.data, self.masked_values, self.masked_pos = self.make_data(data)

    def make_data(self, data):
        # input of size: [n_sample, n_meas * n_nodes, ts]
        batch = []
        masked_values = []
        masked_pos = []

        np.random.seed(0)
        n_pred = int(self.configs.n_pred)
        for i in range(data.shape[0]):
            id = np.arange(data.shape[1])
            np.random.shuffle(id)
            masked_pos.append(id[:n_pred])
            masked_values.append(data[i, id[:n_pred], 0].copy())
            if random.random() < 0:
                data[i, id[:n_pred], 0] = 0
            else:
                data[i, id[:n_pred]] = 0
        return data, masked_values, masked_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.masked_values[idx], self.masked_pos[idx], self.Z

def data_generator(data_path, configs, training_mode):
    batch_size = configs.batch_size

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = load_dataset(train_dataset, configs)
    valid_dataset = load_dataset(valid_dataset, configs)
    test_dataset = load_dataset(test_dataset, configs)

    if train_dataset.__len__() < batch_size:
        batch_size = 16

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)

    return train_loader, valid_loader, test_loader




