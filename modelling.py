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
#define your own mse and set greater_is_better=False

np.random.seed(1)


clean_data_frame=pd.read_csv('clean_tabular_data.csv')
X, y = load_airbnb(clean_data_frame)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5
)
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


    best_hyperparams, best_loss = None, np.inf

    for hyperparams in dict_of_params:
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)

        y_validation_pred = model.predict(X_validation)
        validation_loss_root_mean_squared_error = np.sqrt(mean_squared_error(y_validation, y_validation_pred))
        print(f"H-Params: {hyperparams} Validation loss: {validation_loss_root_mean_squared_error}")
        
        if validation_loss_root_mean_squared_error < best_loss:
            best_loss = validation_loss_root_mean_squared_error
            best_hyperparams = hyperparams
            RMSE_loss=np.sqrt(mean_squared_error(y_validation, y_validation_pred))

            
            
    performance_metrics_dict = {'validation_RMSE': RMSE_loss}
    best_hyperparapeter_values=best_hyperparams
   

    best_model =model_class(**best_hyperparapeter_values)
    best_model.fit(X_train, y_train)#we need to train our best model on unseen data in order to get the metrics
    train_R2=best_model.score(X_train, y_train)
    test_R2=best_model.score(X_test, y_test)
    cv_score=cross_val_score(best_model, X_train, y_train, cv = 10)
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    test_RMSE = mse**(1/2.0)
    
    performance_metrics_dict={}
    return best_model, best_hyperparapeter_values, performance_metrics_dict

hyperparameter_grid= {
    'alpha': 10.0 ** -np.arange(1, 7),
    'loss': ['huber', 'epsilon_insensitive', 'squared_error'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    'max_iter': np.arange(1000,10000,2000)
}

# al=custom_tune_regression_model_hyperparameters(SGDRegressor, X_train=X_train, y_train=y_train, 
#                                                 X_validation=X_validation,y_validation=y_validation,
#                                                 X_test= X_test,y_test= y_test,dict_of_hyperparameters=hyperparameter_grid)
# print(al)

def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_validation ,y_validation, X_test, y_test, parameter_grid):
        
       # mse = make_scorer(mean_squared_error,greater_is_better=False)
        model=model_class()
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=parameter_grid, scoring= 'neg_root_mean_squared_error'
        )
        grid_search.fit(X_train, y_train)
        grid_search.predict(X_validation)
        grid_search.best_params_
        grid_search.best_score_#returns best negative root mean square error

        best_model =model_class(**grid_search.best_params_)
        best_model.fit(X_train, y_train)#we need to train our best model on unseen data in order to get the metrics
        train_R2=best_model.score(X_train, y_train)
        test_R2=best_model.score(X_test, y_test)
        cv_score=cross_val_score(best_model, X_train, y_train, cv = 10)
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        test_RMSE = mse**(1/2.0)
        return  best_model, grid_search.best_params_, {'validation_RMSE':grid_search.best_score_, 
                                                       'train_rsquared': train_R2,
                                                       'test_rsquared': test_R2,
                                                       'mse': mse,
                                                       'test_RMSE': test_RMSE}

tuned_model, best_params, metrics =tune_regression_model_hyperparameters(model_class=SGDRegressor, X_train=X_train, y_train=y_train, 
                                                 X_validation=X_validation,y_validation=y_validation,X_test= X_test,y_test= y_test, 
                                                  parameter_grid=hyperparameter_grid)



def save_model(folder):

    desired_dir = folder
    # json_dict={'hyperparameters.json': best_params, 'metrics.json': metrics}
    # for k, v in json_dict.items():
    #      with open(os.path.join(b'desired_dir', k) , mode='w') as f:
    #                     json.dump(v, f)
    with open(os.path.join(desired_dir, 'hyperparameters.json') , mode='w') as f:
           json.dump(best_params, f)

    with open(os.path.join(desired_dir,'model.joblib' ) , mode='wb') as f:
         joblib.dump(tuned_model, f)

    with open(os.path.join(desired_dir, 'metrics.json') , mode='w') as f:
         json.dump(metrics, f)
            
save_model("models/regression")

def  evaluate_all_models():
    hyperparameter_grids= [ hyperparameter_grid_decision_tree={},hyperparameter_grid_random_forests={}, hyperparameter_grid_gradient_boosting={}]

    model_classes=[DecisionTreeRegressor,  RandomForestRegressor, GradientBoostingRegressor]
    
    for model in model_classes:
        for grid in hyperparameter_grids:
        tuned_model, best_params, metrics =tune_regression_model_hyperparameters(model_class=model, X_train=X_train, y_train=y_train, 
                                                 X_validation=X_validation,y_validation=y_validation,X_test= X_test,y_test= y_test, 
                                                  parameter_grid=hyperparameter_grid)



if __name__ == "__main__" :
     