import glob
import json
import os
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from torchmetrics import R2Score
from tabular_data import load_airbnb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
import datetime
import itertools
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt



class AirbnbNightlyPriceImageDataset(Dataset):# A class to represent the dataset used in the neural network model

    def __init__(self):
        """_summary_: Selects the necessary features and label from the loaded dataset
        """
    
        self.data = pd.read_csv('clean_tabular_data.csv')
        self.features, self.label  = load_airbnb(self.data, 'Price_Night')
        

    def __getitem__(self, idx):

        """ Parameters
            ----------
            idx: type: int
                _description_: The index of the specified sample.

            Returns
            -------
            _type_: torch tensor 
                _description_: Returns a tuple of the features and labels as a tensors 
        """
         #set feautres and labels separately like the following
        features = self.features.iloc[idx]
        features = torch.tensor(features).float()
        label = self.label.iloc[idx]
        label = torch.tensor(label).float()
        return features, label
    
    def __len__(self):
        """_summary_: represents the number of samples in the dataset

        Returns
        -------
        _type_: int
            _description_: method is used to define the length of the dataset. 
        """
        return len(self.features)



def get_random_split(dataset):

    """_summary_: 
    Parameters
    ----------
    idx: type: torch.utils.data.Dataset
    _description_:The dataset to be split.

    Returns
    -------
    _type_: tuple
        _description_: tuple
        A tuple containing three subsets: train_set, validation_set, and test_set.
    """
    
    train_set, test_set = random_split(dataset, [0.7, 0.3])
    train_set, validation_set = random_split(train_set, [0.5, 0.5])
    
    return train_set, validation_set, test_set


# train_dataset, which contains 70% of the original dataset
# val_dataset, which contains 20% of the original dataset
# test_dataset, which contains 10% of the original dataset

def get_data_loader(dataset, batch_size=32):
    """_summary_: Perform random split on the dataset

    Parameters
    ----------
    dataset : _type_: torch.utils.data.Dataset
        _description_: The dataset to be split.
    batch_size : int, optional
        _description_, by default 32

    Returns
    -------
    _type_:  dict
        _description_: A dictionary containing data loaders for train, validation, and test sets.
    """

    train_set, validation_set, test_set = get_random_split(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    data_loader = {"train": train_loader, "validation": validation_loader, "test": test_loader}
    
    return data_loader


# print(dataset.features.shape)
class NeuralNetwork(nn.Module):
    
    def __init__(self, config, input_dim, output_dim):
        """_summary_: Initialize the layers of a neural network model

        Parameters
        ----------
        config : _type_: list
            _description_: A list of dictionaries containing configuration hyperparameters for the model.
        input_dim : _type_:int
            _description_:  The input dimension of the model.
        output_dim : _type_: int
            _description_: The output dimension of the model.
        """

        '''super().__init__() is called to invoke the initialization method of the nn.Module class,
          which is the parent class of NeuralNetwork. By calling super().__init__(), the NeuralNetwork class 
          inherits and initializes the attributes and methods defined in nn.Module.'''
        super().__init__()


        
        width = config[0]['hidden_layer_width']
        depth = config[0]['depth']

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
        """_summary_: Forward pass of the neural network model.

        Parameters
        ----------
        features : _type_: torch.Tensor
            _description_: The input features.

        Returns
        -------
        _type_: torch.Tensor
            _description_: The output of the model. a forward pass on the features
        """
        x = self.layers(features)
        return x

def train(model, data_loader, optimiser, epochs=15):
    """_summary_: Train a neural network model.

    Parameters
    ----------
    model : _type_: torch.nn.Module
        _description_: The neural network model to train.
    data_loader : _type_: dict
        _description_: A dictionary containing data loaders for train, validation, and test sets.
    optimiser : _type_: torch.optim.Optimizer
        _description_: The optimizer used to update the model's parameters.(weights and biases)
    epochs : int, optional
        _description_, The number of epochs to train the model, by default 15.

    Returns
    -------
    _type_: dict
        _description_: A dictionary containing performance metrics of the trained model.
    """

    
    writer = SummaryWriter()
    batch_idx = 0
    batch_idx2 = 0
    pred_time = []
    start_time = time.time()
    

    for epoch in range(epochs):
        for batch in data_loader['train']:
            features, label = batch
            label = torch.unsqueeze(label, 1)
            inference_start_time = time.time()
            prediction = model(features)
            inference_end_time = time.time()
            time_elapsed = inference_end_time - inference_start_time 
            pred_time.append(time_elapsed)
            loss = F.mse_loss(prediction, label)
            R2_train = R2Score()
            R2_train = R2_train(prediction, label)
            RMSE_train = torch.sqrt(loss)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalars(optimiser.__class__.__name__, {"Train_loss": loss.item()}, batch_idx)
            batch_idx += 1
            
        end_time = time.time()
        
        for batch in data_loader['validation']:
            features, label = batch
            label = torch.unsqueeze(label, 1)
            prediction = model(features)
            loss_val = F.mse_loss(prediction, label)
            writer.add_scalars(optimiser.__class__.__name__, {"Val_loss": loss_val.item()}, batch_idx2)
            R2_val = R2Score()
            R2_val = R2_val(prediction, label)
            RMSE_val = torch.sqrt(loss_val)
            batch_idx2 += 1
            
    training_duration = end_time - start_time
    inference_latency = sum(pred_time)/len(pred_time)
    performance_metrics = {'RMSE_Loss_Train': RMSE_train.item(), 'R2_Score_Train': R2_train.item(), 
                           'RMSE_Loss_Validation': RMSE_val.item(), 'R2_Score_Val': R2_val.item(),
                           'training_duration_seconds': training_duration, 'inference_latency_seconds': inference_latency}
    
    return performance_metrics



# l2_reg = torch.tensor(0.)
# for param in model.parameters():
#     if param.requires_grad==True and len(param)>1: # by setting requires grad == True we make sure that we only apply 
#         #regularization to the parameters that are actually being trained as not all params in a neural network are trainable 
#         l2_reg += torch.norm(param, p=2)
# loss = mse_loss + lambda_value * l2_reg
      

def generate_nn_configs():
    """_summary_: Generate configurations for neural network models.

    Returns
    -------
    _type_: list 
        _description_: A list of dictionaries representing different configurations for neural network models.
        Each dictionary contains the following keys:
        - 'optimiser': The name of the optimizer.
        - 'params': A dictionary of optimizer parameters.
        - 'hidden_layer_width': The width of the hidden layers in the neural network.
        - 'depth': The depth (number of hidden layers) in the neural network.
    """

    all_params = []
    optimiser_params = {
    'adadelta_params': {
        'optimiser': 'Adadelta',
        'params': {
        'lr': [1.0, 0.001, 0.0001],
        'rho': [0.9, 0.7, 0.3],
        'weight_decay': [0, 0.5, 1, 1.5],
        },
    },

    'sgd_params' : {
        'optimiser': 'SGD',
        'params':{
        'lr': [0.001, 0.0001],
        'momentum': [0, 0.1, 0.3, 0.7],
        'weight_decay': [0, 0.5, 1, 1.5],
        
    }},

    'adam_params': {
        'optimiser': 'Adam',
        'params': {
        'lr': [0.001, 0.0001],
        'weight_decay': [0, 0.5, 1, 1.5],
        'amsgrad': [True, False],

    }},

    'adagrad_params': {
        'optimiser': 'Adagrad',
        'params': {
        'lr': [0.01, 0.001, 0.0001],
        'lr_decay': [0, 0.1, 0.3, 0.7],
        'weight_decay': [0, 0.5, 1, 1.5],
        
           }
        }
    }


    for optimiser_classes, mp  in optimiser_params.items():
        keys, values = zip(*mp['params'].items())
        params_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
        # all_params.append(params_dict)

        params_dict_with_keys = [{'optimiser': mp['optimiser'], 'params': d, 'hidden_layer_width':5, 'depth': 5} for d in params_dict]
        all_params.extend(params_dict_with_keys)
    
    return all_params



#I then converted the params to yaml files because then we can use them
def convert_all_params_to_yaml(all_params, yaml_file):
    """_summary_

    Parameters
    ----------
    all_params : _type_: list
        _description_: List of dictionaries representing hyperparameter configurations.
    yaml_file : _type_: str
        _description_: Path to the YAML file to write the configurations.
    """
    

    with open(yaml_file, 'w') as f:
        yaml.safe_dump(all_params, f, sort_keys=False, default_flow_style=False)



#This gets the nn config
def get_nn_config(yaml_file):
    """_summary_

    Parameters
    ----------
    yaml_file : _type_: str
        _description_: Path to the YAML file containing the configurations.

    Returns
    -------
    _type_: list
        _description_: List of dictionaries representing the neural network configurations.
    """
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
        
    return config 



def save_model(model, folder, optimiser, performance_metrics, optimiser_params):
    """_summary_

    Parameters
    ----------
    model : _type_: torch.nn.Module
        _description_: The PyTorch model to be saved.
    folder : _type_: str
        _description_: The folder path where the model will be saved.
    optimiser : _type_: torch.optim.Optimizer
        _description_: The optimizer used for training the model.
    performance_metrics : _type_: dict
        _description_: Dictionary containing the performance metrics of the model.
    optimiser_params : _type_: dict
        _description_: Dictionary containing the optimizer parameters.
    """
    
    if not isinstance(model, torch.nn.Module):
        print("Your model is not a Pytorch Module")    
    
    else:
        
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        optimiser_name = optimiser.__class__.__name__
        model_folder = os.path.join(folder, f"{time}_{optimiser_name}")
        os.makedirs(model_folder, exist_ok=True)#exist_ok=True argument ensures that no exception 
        #is raised if the directory already exists. If exist_ok=False (which is the default), 
        # #it would raise a FileExistsError if the directory already exists.
        model_path = os.path.join(model_folder, 'model.pt')
        sd = model.state_dict()
        torch.save(sd, model_path)
        
        with open(os.path.join(model_folder, "hyperparameters.json"), 'w') as fp:
            json.dump(optimiser_params, fp)
        
        with open(os.path.join(model_folder, 'metrics.json'), 'w') as fp:
            json.dump(performance_metrics, fp)  


def get_optimiser(yaml_file, model):
    """
    Parameters
    ----------
    yaml_file : str
        The path to the YAML configuration file.
    model : torch.nn.Module
        The model for which optimizers will be created.

    Returns
    -------
    list
        A list of optimizer instances.

    Raises
    ------
    ValueError
        If an unknown optimizer type is specified in the YAML file.
    """
    all_params = generate_nn_configs()
    convert_all_params_to_yaml(all_params, yaml_file)
    config = get_nn_config(yaml_file)
    optimisers = []
    
    for optimiser in config:
        
        optimiser_type_str = optimiser['optimiser']
        optimiser_params = optimiser['params']

    
        if optimiser_type_str == 'Adadelta':
            optimiser_type = torch.optim.Adadelta
        elif optimiser_type_str == 'SGD':
            optimiser_type = torch.optim.SGD
        elif optimiser_type_str == 'Adam':
            optimiser_type = torch.optim.Adam
        elif optimiser_type_str == 'Adagrad':
            optimiser_type = torch.optim.Adagrad
        else:
            raise ValueError(f"Unknown optimiser type: {optimiser_type_str}")
        
        optimiser_instance = optimiser_type(model.parameters(), **optimiser_params)
        optimisers.append(optimiser_instance)
    
    return optimisers



def find_best_nn(yaml_file, model, folder):
    """_summary_: Find the best neural network model based on performance metrics.

    Parameters
    ----------
    yaml_file : str
        The path to the YAML configuration file.
    model : torch.nn.Module
        The neural network model.
    folder : _type_: str
        _description_: The folder path where the model will be saved.

    Returns
    -------
    tuple
        A tuple containing the best model name, its hyperparameters, and performance metrics.
    """
   
    optimisers = get_optimiser(yaml_file, model)
    
    for optimiser in optimisers:
        dataset = AirbnbNightlyPriceImageDataset()
        data_loader = get_data_loader(dataset)
        performance_metrics = train(model, data_loader, optimiser, epochs=15)
        state_hyper_dict = optimiser.state_dict()
        optimiser_params = state_hyper_dict['param_groups'][0]# all optimiser hyperparameters 
        save_model(model, folder, optimiser, performance_metrics, optimiser_params)
    
    metrics_files = []
    for entry in os.scandir("models/neural_networks/regression"):
        if entry.is_dir():
            metrics_file = os.path.join(entry.path, "metrics.json")
            if os.path.isfile(metrics_file):
                metrics_files.append(metrics_file)

    
    best_rmse = float('inf')
    
    for file in metrics_files:
        f = open(str(file))
        dic_metrics = json.load(f)
        f.close()
        val_rmse_loss = dic_metrics['RMSE_Loss_Validation']

        if val_rmse_loss < best_rmse:
            best_rmse = val_rmse_loss
           # best_name = str(file).split('/')[-2]
            best_name = os.path.basename(os.path.dirname(file))
            print(best_name)

    path = f'models/neural_networks/regression/{best_name}/'
    print(path)
    best_model_directory = 'models/neural_networks/regression/best_model'
    best_model_metrics_path = os.path.join(best_model_directory, 'metrics.json')
    best_model_params_path = os.path.join(best_model_directory, 'hyperparameters.json')
    best_model_path = os.path.join(best_model_directory, 'model.pt')
    
    with open (path + 'hyperparameters.json', 'r') as fp:
        params = json.load(fp)
    
    with open (path + 'metrics.json', 'r') as fp:
        metrics = json.load(fp)
    
    with open(best_model_params_path, 'w') as f:
                json.dump(params, f)
        
    with open(best_model_metrics_path, 'w') as f:
            json.dump(metrics, f)
    
    with open(os.path.join(folder, best_name, 'model.pt'), 'rb') as fp:
        model_state_dict = torch.load(fp)
        torch.save(model_state_dict, best_model_path)

    return best_name, params, metrics    




if __name__ == "__main__" :
   
    torch.manual_seed(2)# to ensure the same output for each run 
#     dataset = AirbnbNightlyPriceImageDataset()
   # model = NeuralNetwork(config , 11, 1)
    dataset = AirbnbNightlyPriceImageDataset()
    config = generate_nn_configs()

    model = NeuralNetwork( config, 11, 1)
# #     best_name, params, metrics = find_best_nn('nn_config.yaml', model, dataset,'models/neural_networks/regression')
# #   #  print_model_info(best_name, metrics, params)
    find_best_nn('nn_config.yaml', model, 'models/neural_networks/regression')
    #print(len(get_optimiser('nn_config.yaml', model)))
    
   


# print(all_params)
    

 