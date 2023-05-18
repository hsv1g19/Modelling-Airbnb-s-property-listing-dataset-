import json
import os
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from tabular_data import load_airbnb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
import datetime
import itertools
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

np.random.seed(42)

class AirbnbNightlyPriceImageDataset(Dataset):

    def __init__(self):
    
        self.data = pd.read_csv('clean_tabular_data.csv')
        self.features, self.label  = load_airbnb(self.data, 'bedrooms')
        category_column = self.data['Category']
        encoder = OneHotEncoder()
        
        # Fit the encoder to your data
        encoder.fit(category_column.values.reshape(-1, 1))
        category_column = encoder.transform(category_column.values.reshape(-1, 1))
        pd.concat([self.features, pd.DataFrame(category_column.toarray())], axis=1)  # Concatenate one-hot encoded category
        #When axis=1, it means that the concatenation is done horizontally or column-wise.
        

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
    
    def __init__(self, config , input_dim, output_dim):
        super().__init__()
        
        width = config['hidden_layer_width']
        depth = config['depth']


        # input layer
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        
        # hidden layers
        for hidden_layer in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        
        # output layer
        layers.append(nn.Linear(width, output_dim))
        
        # create a sequential model
        self.layers = nn.Sequential(*layers)
        
    def forward(self, features):
        x = self.layers(features)
        return x




def train(model, train_loader,val_loader, epochs, hyperparam_dict, lambda_value = 0.001):

    optimiser_class = hyperparam_dict["optimiser"]
    optimiser_instance = getattr(torch.optim, optimiser_class)
    optimiser = optimiser_instance(model.parameters(), lr=hyperparam_dict["learning_rate"], weight_decay=1e-5)    
    writer = SummaryWriter()
    batch_idx = 0
    total_inference_time = 0
    train_start_time = time.time()
    pred_time = []
    for epoch in range(epochs):
        for batch in train_loader:
            X_train, y_train = batch
            y_train = y_train.float()
            y_train = torch.unsqueeze(y_train, 1)

            inference_start_time = time.time()
            prediction = model(X_train)
            inference_end_time = time.time()
            
            time_elapsed = inference_end_time - inference_start_time
            pred_time.append(time_elapsed)
        
            mse_loss = F.mse_loss(prediction, y_train)
            # l2_reg = torch.tensor(0.)
            # for param in model.parameters():
            #     if param.requires_grad==True and len(param)>1: # by setting requires grad == True we make sure that we only apply 
            #         #regularization to the parameters that are actually being trained as not all params in a neural network are trainable 
            #         l2_reg += torch.norm(param, p=2)
            # loss = mse_loss + lambda_value * l2_reg
            mse_loss.backward()
          #  print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', mse_loss.item(), batch_idx )
            batch_idx +=1
            


        train_end_time = time.time()
        
        for val_batch in val_loader:
            X_val, y_val = val_batch
            y_val = y_val.float()
            y_val = torch.unsqueeze(y_val, 1)
            val_prediction = model(X_val)
            val_loss = F.mse_loss(val_prediction, y_val)
            writer.add_scalar('val_loss', val_loss.item(), batch_idx)

    train_time = train_end_time - train_start_time
    inference_latency = sum(pred_time)/len(pred_time)
   


    return inference_latency, train_time
    

def evaluate_model(model):
    full_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    X_train, y_train = next(iter(full_train_loader))
    y_train = y_train.float()
    
    full_val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    X_val, y_val = next(iter(full_val_loader))
    y_val = y_val.float()

    full_test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    X_test, y_test = next(iter(full_test_loader))
    y_test = y_test.float()
    

    y_train = torch.unsqueeze(y_train, 1)
    y_test = torch.unsqueeze(y_test, 1)
    y_val = torch.unsqueeze(y_val, 1)


    y_train_pred = model(X_train)
    train_rmse_loss = torch.sqrt(F.mse_loss(y_train_pred, y_train))
    train_r2_score = 1 - train_rmse_loss / torch.var(y_train)

    print('Train_RMSE: ', train_rmse_loss.item())
    print('Train_R2: ', train_r2_score.item())

    y_test_pred = model(X_test)
    test_rmse_loss = torch.sqrt(F.mse_loss(y_test_pred, y_test))
    test_r2_score = 1 - test_rmse_loss / torch.var(y_test)
    
    print('Test_RMSE: ', test_rmse_loss.item())
    print('Test_R2: ', test_r2_score.item())

    y_validation_pred = model(X_val)
    validation_rmse_loss = torch.sqrt(F.mse_loss(y_validation_pred, y_val))
    validation_r2_score = 1 - validation_rmse_loss / torch.var(y_val)

    print('validation_RMSE: ', validation_rmse_loss.item())
    print('validation_R2: ', validation_r2_score.item())

    metrics = {}
    RMSE_loss = [train_rmse_loss, validation_rmse_loss, test_rmse_loss]
    R_squared = [train_r2_score, validation_r2_score, test_r2_score]

    metrics["RMSE_loss"] = [loss.item() for loss in RMSE_loss]
    metrics["R_squared"] = [score.item() for score in R_squared]
    
    return metrics

def plots():
      # Convert tensors to numpy arrays
    y_train = y_train.numpy()
    y_val = y_val.numpy()
    y_test = y_test.numpy()
    y_train_pred = y_train_pred.detach().numpy()
    y_validation_pred = y_validation_pred.detach().numpy()
    y_test_pred = y_test_pred.detach().numpy()

    # Plot predictions vs. true values
    fig, ax = plt.subplots()
    ax.scatter(y_train, y_train_pred, color='blue', label='Train')
    ax.scatter(y_val, y_validation_pred, color='green', label='Validation')
    ax.scatter(y_test, y_test_pred, color='red', label='Test')
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Predictions vs. True Values')
    ax.legend()
    plt.show()
#train(model, train_loader=train_loader, val_loader=val_loader, epochs=10)

def get_nn_config():

    with open("nn_config.yaml", "r") as f:
        hyper_dict = yaml.safe_load(f)

    return hyper_dict



def save_model(model, hyper_dict, metrics):
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = f"models/neural_networks/regression/{current_time}"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if isinstance(model, torch.nn.Module):
        model_path = os.path.join(model_dir, "model.pt")
        hyperparams_path = os.path.join(model_dir, "hyperparameters.json")
        metrics_path = os.path.join(model_dir, "metrics.json")

    # save the model
        torch.save(model.state_dict(), model_path)
        # metrics_serializable = {
        #     key: {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in sub_dict.items()}
        #     for key, sub_dict in metrics.items()
        # }
        
        with open(hyperparams_path, 'w') as f:
                json.dump(hyper_dict, f)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

def generate_nn_configs():
    configs = {'optimiser': ['SGD','Adam'],
                'learning_rate': [0.001, 0.01],
                'hidden_layer_width': [3, 10],
                'depth': [3, 12]}

    keys, values = zip(*configs.items())
    dict_of_params= (dict(zip(keys, v)) for v in itertools.product(*values))


    hyperparam_list = []
    for hyperparams in dict_of_params:
        hyperparam_list.append(hyperparams)

    return hyperparam_list

    #16 different parametrisations 
def find_best_nn():
    configs = generate_nn_configs()
    best_model = None
    best_r2 = -float('inf')
    best_rmse = float('inf')

    for config in configs:
        model = FcNet(config=config, input_dim=11, output_dim = 1)
        train( model,train_loader,val_loader, epochs = 10, hyperparam_dict = config, lambda_value = 0.001)
        metrics = evaluate_model(model)
        save_model(model=model,hyper_dict = config, metrics= metrics)
        val_rmse_loss= metrics["RMSE_loss"][1]
        validation_r2=metrics['R_squared'][1]

        if val_rmse_loss < best_rmse and validation_r2 > best_r2:
                best_model = model
                best_r2 = validation_r2
                best_rmse = val_rmse_loss
    
    model_directory = 'models/neural_networks/regression/best_model'
    best_model_config = config
    best_model = model
    best_model_metrics = metrics
    
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    if isinstance(model, torch.nn.Module):
        best_model_metrics_path = os.path.join(model_directory, 'metrics.json')
        best_model_params_path = os.path.join(model_directory, 'hyperparameters.json')
        best_model_path = os.path.join(model_directory, 'model.pt')
        
    # save the model
        torch.save(model.state_dict(), best_model_path)
        # metrics_serializable = {
        #     key: {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in sub_dict.items()}
        #     for key, sub_dict in metrics.items()
        # }
        
        with open(best_model_params_path, 'w') as f:
                json.dump(config, f)
        
        with open(best_model_metrics_path, 'w') as f:
            json.dump(metrics, f)

    
    return best_model, best_model_metrics, best_model_config



if __name__ == "__main__" :
   
   
    find_best_nn()
   
 #  model = FcNet(config= get_nn_config(), input_dim = 11, output_dim =1 )
  #  avg_inference_latency, train_time =train(model, train_loader, val_loader, epochs = 10, hyperparam_dict=hyperparam_dict)
    #sd = model.state_dict()
   # print(sd)
    # torch.save(model.state_dict(), 'model.pt')
    # state_dict = torch.load('model.pt')
    # new_model = FcNet()
    # new_model.load_state_dict(state_dict)
    # train(new_model)
  #  print(avg_inference_latency)
   # print(find_best_nn())
#   print(train(model =model, train_loader=train_loader, val_loader=val_loader, epochs = 10, hyperparam_dict=get_nn_config()))
   #metrics = evaluate_model(model=model)
   #save_model(model=model, hyper_dict=get_nn_config(), metrics=metrics)
 #  save_model(model=, hyper_dict=, metrics=)
    

 