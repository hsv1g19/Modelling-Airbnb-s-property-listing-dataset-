import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from tabular_data import load_airbnb
import torch.nn as nn

class AirbnbNightlyPriceImageDataset(Dataset):

    def __init__(self):
    
        self.data = pd.read_csv('clean_tabular_data.csv')
        self.features, self.label  = load_airbnb(self.data, 'Price_Night')

    def __getitem__(self, idx):
    
        return (torch.tensor(self.features[idx]), self.label[idx])

    def __len__(self):
        return len(self.features)

dataset = AirbnbNightlyPriceImageDataset()


train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

# train_dataset, which contains 70% of the original dataset
# val_dataset, which contains 20% of the original dataset
# test_dataset, which contains 10% of the original dataset

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

#example = next(iter(train_loader))

import itertools

for idx in itertools.islice(range(len(train_dataset)), 5):
    print("Index: ", idx)
    print("Features shape: ", train_dataset[idx][0].shape)
    print("Label: ", train_dataset[idx][1])
# for sample in itertools.islice(train_dataset, 5):
#     print(sample)

# class NN():
    
#     def __init__(self):
#         self.linear_layer = torch.nn.Linear(11, 1)
#         pass  

#     def __call__(self, features) :
        
#         return self.linear_layer(features)
