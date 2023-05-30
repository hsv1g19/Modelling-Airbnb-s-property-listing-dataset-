import pickle
from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import typing
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import json
import joblib
import os
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#define your own mse and set greater_is_better=False




clean_data_frame=pd.read_csv('clean_tabular_data.csv')#read in csv file saved in the same folder is modelling.py file 
X, y = load_airbnb(clean_data_frame, 'Price_Night')# let the label be price per night of listing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5
)#this means 70% is training set, 15% is validation set, 15% is the test set


scaler = StandardScaler()#scale data for each dataset/ normalise them 
X_train= scaler.fit_transform(X_train)
X_test= scaler.fit_transform(X_test)
X_validation= scaler.fit_transform(X_validation)


#print(X_train.shape, y_train.shape)
# At this point, it should only include numerical values. I will use the other modes of data later.
# Used sklearn to train a linear regression model to predict the "Price_Night" feature from the tabular data. 
# using built in model class SGDRegressor
sgdr = SGDRegressor()
train_model = sgdr.fit(X_train, y_train)
y_test_pred = sgdr.predict(X_test)
y_train_pred = sgdr.predict(X_train)
#y_validation_pred = sgdr.predict(X_validation)#check how good the model is through the validation prediction and accuracy test

# Using sklearn I compute the key measures of performance for your regression model. That should include the RMSE, and R^2 for both 
# the training and test sets to compare
# to the models you will train next.

print("RMSE (scikit-learn):", mean_squared_error(y_test, y_test_pred, squared=False))
print("R2 (scikit-learn):", r2_score(y_test, y_test_pred))

print("RMSE (scikit-learn):", mean_squared_error(y_train, y_train_pred, squared=False))
print("R2 (scikit-learn):", r2_score(y_train, y_train_pred))

def custom_tune_regression_model_hyperparameters(model_class,  X_train, y_train, X_validation ,y_validation, X_test, y_test, dict_of_hyperparameters: typing.Dict[str, typing.Iterable]):
    """_summary_

    Parameters
    ----------
    model_class :  The class of regression model to be tuned
        _description_
    X_train : _type_: numpy.ndarray
        _description_: The normalized training set for the features
    y_train : _type_: numpy.ndarray
        _description_: The normalized training set for the label
    X_validation : _type_: numpy.ndarray
        _description_: The normalized validation set for the features
    y_validation : _type_: numpy.ndarray
        _description_: The normalized validation set for the label
    X_test : _type_:numpy.ndarray
        _description_: The normalized test set for the features
    y_test : _type_:numpy.ndarray
        _description_: The normalized test set for the label
    dict_of_hyperparameters : typing.Dict[str, typing.Iterable]
        _description_: A dictionary of specified hyperparameters corresponding to the model_class.

    Returns
    -------
    best_model:
        The best model_class.
    best_params: dict
        A dictionary containing the best parameters of the best model_class.
    metrics: dict
        A dictionary of the performance metrics of the best model_class.    
    """
        

    keys, values = zip(*dict_of_hyperparameters.items())# zip the items so they correspond to the correct key value pairs and don't get mixed up
    dict_of_params= (dict(zip(keys, v)) for v in itertools.product(*values))# all possible combinations of values within each dictionary for each model class


    best_hyperparams= None
    best_val_rmse =  np.inf
    best_model = None
    metrics={}

    for hyperparams in dict_of_params:
         # Create model object with given hyperparameters
        model = model_class(**hyperparams)

        # Train the model on the training set
        model.fit(X_train, y_train)
        
         # Evaluate the model on the validation set
        y_val_pred = model.predict(X_validation)
        val_rmse= np.sqrt(mean_squared_error(y_validation, y_val_pred))#rmse on validation set
        val_r2 = r2_score(y_validation, y_val_pred)# 
        
        # R2 score, also known as the coefficient of determination, is a statistical measure that 
        # represents the proportion of the variance in the dependent variable that can be explained 
        # by the independent variables in a regression model. It is used to evaluate the goodness-of-fit 
        # of a regression model and can range from negative infinity to 1. A higher R2 score indicates a better fit of the model to the data.
       # print(f"H-Params: {hyperparams} Validation loss: {val_rmse}")

    #    RMSE provides a measure of how well the model's predictions align with the actual values. It is expressed 
    #     in the same units as the target variable, which makes it easily interpretable. Lower RMSE values indicate better performance, 
    #     indicating that the model's predictions are closer to the actual values.
            
        if val_rmse < best_val_rmse:
            best_val_rmse=val_rmse
            best_hyperparams = hyperparams
            best_model = model
            metrics['validation_RMSE'] = val_rmse
            metrics["validation_R2"] = val_r2


    best_model.fit(X_train, y_train)#retrain the training data on the best model we need to train our best model on unseen data in order to get the metrics
    y_pred = best_model.predict(X_test)
    metrics['test_RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['test_R2'] = best_model.score(X_test, y_test)
    metrics['train_R2'] = best_model.score(X_train, y_train)
    #best_model =model_class(**best_hyperparams)
   
  #  cv_score=cross_val_score(best_model, X_train, y_train, cv = 10)
   
    metrics['mean_absolute_error']=mean_absolute_error(y_test, y_pred)
   
   
    return best_model, best_hyperparams, metrics


def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_validation ,y_validation, X_test, y_test, parameter_grid):
        """_summary_

    Parameters
        ----------
        model_class :  The class of regression model to be tuned
            _description_
        X_train : _type_: numpy.ndarray
            _description_: The normalized training set for the features
        y_train : _type_: numpy.ndarray
            _description_: The normalized training set for the label
        X_validation : _type_: numpy.ndarray
            _description_: The normalized validation set for the features
        y_validation : _type_: numpy.ndarray
            _description_: The normalized validation set for the label
        X_test : _type_:numpy.ndarray
            _description_: The normalized test set for the features
        y_test : _type_:numpy.ndarray
            _description_: The normalized test set for the label
        parameter_grid : _type_: dictionary
            _description_: A dictionary of specified hyperparameters corresponding to the model_class.

        Returns
        -------
        best_model:
            The best model_class.
        best_params: dictionary
            A dictionary containing the best parameters of the best model_class.
        metrics: dictionary
            A dictionary of the performance metrics of the best model_class.    
        """


       # mse = make_scorer(mean_squared_error,greater_is_better=False)
        model=model_class()
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=parameter_grid, scoring= 'neg_root_mean_squared_error',cv=5
        )# 5 folds cross validation 
        grid_search.fit(X_train, y_train)
        grid_search.predict(X_validation)
        grid_search.best_params_#returns the best_params for validation data set
        grid_search.best_score_#returns best negative root mean square error for validation dataset

        best_model =model_class(**grid_search.best_params_)# the same as best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)#we need to train our best model on unseen data in order to get the metrics
        y_train_pred= best_model.predict(X_train)
        y_validation_pred = best_model.predict(X_validation)

        train_R2=best_model.score(X_train, y_train)
        validation_mse =  mean_squared_error(y_validation, y_validation_pred)
        validation_rmse = validation_mse**(1/2)
        validation_R2=best_model.score(X_validation, y_validation)
        train_R2=best_model.score(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        #cv_score=cross_val_score(best_model, X_train, y_train, cv = 10)
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        test_RMSE = mse**(1/2.0)
        #mae = mean_absolute_error(y_test, y_pred)# mean absolute error
        metrics = {'avg-kflod_validation_rmse' :grid_search.best_score_, 
                                                       'validation_rmse': validation_rmse,
                                                       'train_rsquared': train_R2,
                                                       'validation_rsquared': validation_R2,
                                                       'train_rmse': train_rmse
                                                       }

        return  best_model, grid_search.best_params_, metrics


def convert(o):
    #this function is used to convert specific NumPy integer types to float, and it raises an error for other types. 
        
        if isinstance(o, np.int64) or isinstance(o, np.int32): return float(o)  
        raise TypeError

def save_model(folder, best_params, tuned_model, metrics):
    """_summary_: saves the best model, best hyperparams and best metrics in respective folders 

    Parameters
    ----------
    folder : _type_: string
        _description_
    best_params : _type_
        _description_:  A dictionary containing the best parameters of the best model_class.
    tuned_model : _type_
        _description_:  The best model with after finding its optimum hyperparameters
    metrics : _type_: dictionary 
        _description_: A dictionary of the performance metrics of the best model_class. 
    """

    desired_dir = folder

    # The default=convert argument is used to specify a function (convert) that will be called for objects that 
    # are not serializable by default in JSON. In this case, the convert function is used to handle the conversion 
    # of NumPy integer types to floats
    with open(os.path.join(desired_dir, 'hyperparameters.json') , mode='w') as f:
           json.dump(best_params, f, default=convert)

    with open(os.path.join(desired_dir,'model.joblib' ) , mode='wb') as f:
         joblib.dump(tuned_model, f)

    with open(os.path.join(desired_dir, 'metrics.json') , mode='w') as f:
         json.dump(metrics, f)
            


def  evaluate_all_models(task_folder = 'models/regression'):
    """_summary_: Evaluate and tune hyperparameters for multiple regression models. The loop iterated over the dict and saves each tuned model

    Parameters
    ----------
    task_folder : str
        _description_, by default 'models/regression' which is the derectory the models will be saved
    """
    model_params = {
         'decision_tree':{
         'model':DecisionTreeRegressor,
         'params':{"splitter":["best","random"],
           "max_depth" : np.arange(1,13,3),
           "min_samples_leaf": np.arange(0.1, 1.0, 0.18),
           "min_weight_fraction_leaf":np.arange(0,0.5,0.1),
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90],
           "ccp_alpha": [0.001, 0.01, 0.1]
           }
         },
         'random_forest':{
         'model':RandomForestRegressor,
         'params' : {
         "max_features" : ["sqrt", "log2", None], # Number of features to consider at every split
         "max_depth" : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],# Maximum number of levels in tree
         "min_samples_split" : np.linspace(0.1, 1.0, 5, endpoint=True),# Minimum number of samples required to split a node
         "min_samples_leaf" : np.linspace(0.1, 1.0, 5, endpoint=True),# Minimum number of samples required at each leaf node
         "bootstrap": [True, False],# Method of selecting samples for training each tree
         "ccp_alpha": [0.001, 0.01, 0.1]  }
         },
         'gradient_boosting' : {
         'model': GradientBoostingRegressor,
         'params': { 
         'learning_rate': [0.01,0.1,1],
           'n_estimators' : [1, 8 ,16, 32, 64, 100],
           'max_depth': [20, 30, 100, None],
           'min_samples_split' : np.linspace(0.1, 1.0, 5, endpoint=True),
           'min_samples_leaf' : np.linspace(0.1, 0.5, 4, endpoint=True),
           'max_features': ['auto', 'sqrt', 'log2'],
           'min_impurity_decrease': [0.001, 0.01, 0.1]
         
                 }
        },
        'sgdrregressor': {
         'model': SGDRegressor,
         'params':{
         'alpha': 10.0 ** -np.arange(1, 7),
        'loss': ['huber', 'epsilon_insensitive', 'squared_error'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'max_iter': [100,1000,10000]
            
                }
             }
         }
         
    # scores=[]
    for model_classes, mp in model_params.items():
        tuned_model, best_params, metrics =tune_regression_model_hyperparameters(model_class=mp['model'], X_train=X_train, y_train=y_train, 
                                                 X_validation=X_validation,y_validation=y_validation,X_test= X_test,y_test= y_test, 
                                                  parameter_grid=mp['params'])
        # scores.append({
        #      'tuned_model': model_classes,
        #      'best_params': best_params,
        #      'metrics': metrics
        #})
        save_model(folder=f'{task_folder}/{model_classes}', tuned_model=tuned_model, best_params=best_params, metrics=metrics)
    
    
        

def find_best_model(models_directory):

    """_summary_: 
    this fucntion takes the models_directory as input, which is the directory where the saved models are stored.
    It initializes variables to keep track of the best model, best R-squared (validation_r2), and best RMSE (avg_kfold_val_rmse). 
    Initially, the best R-squared is set to negative infinity, and the best RMSE is set to positive infinity.
    It iterates through each model in the models_directory and reads the metrics from the corresponding metrics file (metrics.json).
    It compares the average k-fold validation RMSE (avg_kfold_val_rmse) and validation R-squared (validation_r2) of the current model with the
    best RMSE and best R-squared values obtained so far. 
    If the current model has a lower RMSE and a higher R-squared, it becomes the new best model.
    After iterating through all the models, it identifies the path of the best model's metrics file, hyperparameters file, and model file.
    It loads the metrics, hyperparameters, and model using the identified paths.
    Finally, it returns the optimum metrics, optimum hyperparameters, and optimum model.
    """

    # the code below commented out shows an alternative approck to finding the best model 
        #  with open('data.json') as data_file:
        #     data = json.load(data_file)
        #     for v in data.values():
        #          print(v['x'], v['y'], v['yr'])
    # with open(os.path.join(desired_dir, 'hyperparameters.json') , mode='w') as f:
    #        json.dump(best_params, f, default=convert)
    # for file_name in os.listdir('models/regression/decision_tree'):
    #      with open(file_name, mode='r') as f:
    #           print(file_name)
    # minvalrmse=[]
    # maxvalr2=[]
    best_model = None
    best_r2 = -float('inf')
    best_rmse = float('inf')
    models_directory_list=os.listdir(models_directory)
    for model_name in models_directory_list:
        metrics_path = os.path.join(models_directory, model_name, 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)
            #val_r2 = metrics['validation_r2'] 
        avg_kfold_val_rmse = metrics["avg-kflod_validation_rmse"]
        validation_r2=metrics['validation_rsquared']

        if avg_kfold_val_rmse < best_rmse and validation_r2 > best_r2:
            best_model = model_name
            best_r2 = validation_r2
            best_rmse = avg_kfold_val_rmse

    best_model_metrics_path = os.path.join(models_directory, best_model, 'metrics.json')
    best_model_params_path = os.path.join(models_directory, best_model, 'hyperparameters.json')
    best_model_path = os.path.join(models_directory, best_model, 'model.joblib')

    with open(best_model_metrics_path) as f:
        optimum_metrics = json.load(f)
    with open(best_model_params_path) as f:
        optimum_params = json.load(f)
    with open(best_model_path, mode='rb') as f:
         optimum_model = joblib.load(f)
    
   
    with open(os.path.join('models/best_regression_model','model.joblib' ) , mode='wb') as f:
         joblib.dump(optimum_model, f)

    return optimum_metrics, optimum_params, optimum_model
    #         maxvalr2.append(validation_r2)
    # #print(min(minvalrmse))
    # model_to_load=models_directory_list[np.argmax(maxvalr2)]
    # model_path = os.path.join(models_directory, model_to_load, 'metrics.json')
    #     with open(metrics_path) as f:

def visualise_graphs(folder, X_test, y_test):
    """_summary_: plots the graphs of the predicted and actual labels against each sample as well as the residuals 

    Parameters
    ----------
    folder : _type_: str
        _description_: the directory to find the best model
    X_test : _type_:numpy.ndarray
            _description_: The normalized test set for the features
    y_test : _type_:numpy.ndarray
            _description_: The normalized test set for the label
    """
    best_model_path = os.path.join(folder, 'model.joblib')
    with open(best_model_path, mode='rb') as f:
       best_model = joblib.load(f)
        
    y_pred = best_model.predict(X_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Plotting predictions
    plt.figure()
    plt.scatter(range(len(y_test)), y_test, color='blue', label='True')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.show()

    # Plotting residuals
    plt.figure()
    plt.scatter(range(len(y_test)), residuals, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.title('Residual Plot')
    plt.show()

    # Calculate and print the mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)
                
                
         


if __name__ == "__main__" :
    np.random.seed(42)# ensures each run gives the same output
    visualise_graphs('models/best_regression_model', X_test, y_test)


    #  save_model("models/regression")
    #evaluate_all_models("models/regression")
    #print(type(X_validation))
      