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
#define your own mse and set greater_is_better=False

np.random.seed(42)


clean_data_frame=pd.read_csv('clean_tabular_data.csv')
X, y = load_airbnb(clean_data_frame, 'Price_Night')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5
)#this means 70% is training set, 15% is validation set, 15% is the test set


scaler = StandardScaler()#scale data for each dataset
X_train= scaler.fit_transform(X_train)
X_test= scaler.fit_transform(X_test)
X_validation= scaler.fit_transform(X_validation)


#print(X_train.shape, y_train.shape)
sgdr = SGDRegressor()
train_model = sgdr.fit(X_train, y_train)
y_test_pred = sgdr.predict(X_test)

y_validation_pred = sgdr.predict(X_validation)#check how good the model is through the validation prediction and accuracy test


# print('this is the accuracy score:', accuracy_score(y_validation, y_validation_pred))

# print("RMSE (scikit-learn):", mean_squared_error(y_test, y_test_pred, squared=False))
# print("R2 (scikit-learn):", r2_score(y_test, y_test_pred))

# print("RMSE (scikit-learn):", mean_squared_error(y_train, y_test_pred, squared=False))
# print("R2 (scikit-learn):", r2_score(y_train, y_test_pred))

def custom_tune_regression_model_hyperparameters(model_class,  X_train, y_train, X_validation ,y_validation, X_test, y_test, dict_of_hyperparameters: typing.Dict[str, typing.Iterable]):

    keys, values = zip(*dict_of_hyperparameters.items())
    dict_of_params= (dict(zip(keys, v)) for v in itertools.product(*values))


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
        val_r2 = r2_score(y_validation, y_val_pred)
       # print(f"H-Params: {hyperparams} Validation loss: {val_rmse}")
        
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
        
       # mse = make_scorer(mean_squared_error,greater_is_better=False)
        model=model_class()
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=parameter_grid, scoring= 'neg_root_mean_squared_error',cv=5
        )# 5 kfolds cross validation 
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
        #mae = mean_absolute_error(y_test, y_pred)
        return  best_model, grid_search.best_params_, {'avg-kflod_validation_rmse' :grid_search.best_score_, 
                                                       'validation_rmse': validation_rmse,
                                                       'train_rsquared': train_R2,
                                                       'validation_rsquared': validation_R2,
                                                       'train_rmse': train_rmse
                                                       }

# tuned_model, best_params, metrics =tune_regression_model_hyperparameters(model_class=SGDRegressor, X_train=X_train, y_train=y_train, 
#                                                 X_validation=X_validation,y_validation=y_validation,X_test= X_test,y_test= y_test, 
#                                                  parameter_grid=hyperparameter_grid)
def convert(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32): return float(o)  
        raise TypeError

def save_model(folder, best_params, tuned_model, metrics):

    desired_dir = folder
    # json_dict={'hyperparameters.json': best_params, 'metrics.json': metrics}
    # for k, v in json_dict.items():
    #      with open(os.path.join(b'desired_dir', k) , mode='w') as f:
    #                     json.dump(v, f)
    with open(os.path.join(desired_dir, 'hyperparameters.json') , mode='w') as f:
           json.dump(best_params, f, default=convert)

    with open(os.path.join(desired_dir,'model.joblib' ) , mode='wb') as f:
         joblib.dump(tuned_model, f)

    with open(os.path.join(desired_dir, 'metrics.json') , mode='w') as f:
         json.dump(metrics, f)
            


def  evaluate_all_models(task_folder = 'models/regression'):

    model_params = {
         'decision_tree':{
         'model':DecisionTreeRegressor,
         'params':{"splitter":["best","random"],
           "max_depth" : np.arange(1,13,3),
           "min_samples_leaf": np.arange(0.1, 1.0, 0.18),
           "min_weight_fraction_leaf":np.arange(0,0.5,0.1),
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
           }
         },
         'random_forest':{
         'model':RandomForestRegressor,
         'params' : {
         "max_features" : ["sqrt", "log2", None], # Number of features to consider at every split
         "max_depth" : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],# Maximum number of levels in tree
         "min_samples_split" : np.linspace(0.1, 1.0, 5, endpoint=True),# Minimum number of samples required to split a node
         "min_samples_leaf" : np.linspace(0.1, 1.0, 5, endpoint=True),# Minimum number of samples required at each leaf node
         "bootstrap": [True, False]# Method of selecting samples for training each tree
            }
         },
         'gradient_boosting' : {
         'model': GradientBoostingRegressor,
         'params': { 
         'learning_rate': [0.01,0.1,1],
           'n_estimators' : [1, 8 ,16, 32, 64, 100],
           'max_depth': [20, 30, 100, None],
           'min_samples_split' : np.linspace(0.1, 1.0, 5, endpoint=True),
           'min_samples_leaf' : np.linspace(0.1, 0.5, 4, endpoint=True),
           'max_features': ['auto', 'sqrt', 'log2']
         
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
            avg_kfold_val_rmse = metrics['avg-kflod_validation_rmse']
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

    return optimum_metrics, optimum_params, optimum_model
    #         maxvalr2.append(validation_r2)
    # #print(min(minvalrmse))
    # model_to_load=models_directory_list[np.argmax(maxvalr2)]
    # model_path = os.path.join(models_directory, model_to_load, 'metrics.json')
    #     with open(metrics_path) as f:

                
                
         


if __name__ == "__main__" :
     
   # print(find_best_model('models/regression'))
    
  #   tuned_model, best_params, metrics =tune_regression_model_hyperparameters(model_class=SGDRegressor, X_train=X_train, y_train=y_train, 
                     #                            X_validation=X_validation,y_validation=y_validation,X_test= X_test,y_test= y_test, 
                  #                                parameter_grid=hyperparameter_grid)


    #  save_model("models/regression")
    # evaluate_all_models()
    print(X.shape)
      