import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from tabular_data import load_airbnb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml


class AirbnbNightlyPriceImageDataset(Dataset):

    def __init__(self):
    
        self.data = pd.read_csv('clean_tabular_data.csv')
        self.features, self.label  = load_airbnb(self.data, 'Price_Night')

    def __getitem__(self, idx):
    
        return  (torch.tensor(self.features.iloc[idx]).float(), self.label[idx])

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


#print(dataset.features.shape, type(dataset.features))
# print(dataset.label.shape, type(dataset.label.shape))

# import itertools

# for idx in itertools.islice(range(len(train_dataset)), 5):
#     print("Index: ", idx)
#     print("Features shape: ", train_dataset[idx][0].shape)
#     print("Label: ", train_dataset[idx][1])
# # for sample in itertools.islice(train_dataset, 5):
# #     print(sample)
example = next(iter(train_loader))#gets a single batch
X_train, y_train = example
#print(example)
# X_train = X_train.float()
# y_train = y_train.float()


# print(dataset.features.shape)
class FcNet(nn.Module):
    
    def __init__(self, input_dim=11, output_dim=1, hidden_dim=10, depth=3):
        super().__init__()
        
        # input layer
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        
        # hidden layers
        for hidden_layer in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # create a sequential model
        self.layers = nn.Sequential(*layers)
        
    def forward(self, features):
        x = self.layers(features)
        return x



model = FcNet()
print(len(model(features=X_train)))



def train(model, train_loader,val_loader, epochs, hyperparam_dict):

    optimizer_class = hyperparam_dict["optimizer"]
    optimizer_instance = getattr(torch.optim, optimizer_class)
    optimizer = optimizer_instance(model.parameters(), lr=hyperparam_dict["learning_rate"])

    
    writer = SummaryWriter()

    batch_idx = 0

    for epoch in range(epochs):
        for batch in train_loader:
            X_train, y_train = batch
            y_train = y_train.float()
            prediction = model(X_train)
            loss = F.mse_loss(prediction, y_train)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx )
            batch_idx +=1
        
        for val_batch in val_loader:
            X_val, y_val = val_batch
            y_val = y_val.float()
            val_prediction = model(X_val)
            val_loss = F.mse_loss(val_prediction, y_val)
            writer.add_scalar('val_loss', val_loss.item(), batch_idx)


    
#train(model, train_loader=train_loader, val_loader=val_loader, epochs=10)

def get_nn_config():
    with open("nn_config.yaml", "r") as f:
        hyper_dict = yaml.safe_load(f)

    return hyper_dict


print(get_nn_config())