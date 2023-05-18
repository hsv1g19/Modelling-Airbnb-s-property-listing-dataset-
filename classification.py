from tabular_data import load_airbnb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from modelling import save_model 
import json
import joblib
import os

np.random.seed(42)

clean_data_frame=pd.read_csv('clean_tabular_data.csv')
X, y = load_airbnb(clean_data_frame, label='Category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5
)#this means 70% is training set, 15% is validation set, 15% is the test set


scaler = StandardScaler()#scale data for each dataset
X_train= scaler.fit_transform(X_train)
X_test= scaler.fit_transform(X_test)
X_validation= scaler.fit_transform(X_validation)


# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)




print("Accuracy_test:", accuracy_score(y_test, y_pred_test))
print("Precision_test:", precision_score(y_test, y_pred_test, average="macro"))
print("Recall_test:", recall_score(y_test, y_pred_test, average="macro"))
print("F1 score_test:", f1_score(y_test,y_pred_test, average="macro"))


print("Accuracy_train:", accuracy_score(y_train, y_pred_train))
print("Precision_train:", precision_score(y_train, y_pred_train, average="macro"))
print("Recall_train:", recall_score(y_train, y_pred_train, average="macro"))
print("F1 score_train:", f1_score(y_train, y_pred_train, average="macro"))


def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_validation ,y_validation, X_test, y_test, parameter_grid):
        

        model=model_class()
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=parameter_grid, scoring= 'accuracy',cv=5
        )# 5 folds cross validation 
        grid_search.fit(X_train, y_train)
        grid_search.predict(X_validation)
        grid_search.best_params_#returns the best_params for validation data set
        grid_search.best_score_#returns model with highest accuracy for validation dataset

        best_model =model_class(**grid_search.best_params_)# the same as best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)#we need to train our best model on unseen data in order to get the metrics
        y_train_pred= best_model.predict(X_train)
        y_validation_pred = best_model.predict(X_validation)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        validation_accuracy = accuracy_score(y_validation, y_validation_pred)
        Precision_validation= precision_score(y_validation, y_validation_pred, average="macro")
        Recall_validation= recall_score(y_validation, y_validation_pred, average="macro")
        F1_score_validation = f1_score(y_validation, y_validation_pred, average="macro")


        return  best_model, grid_search.best_params_, {'avg-kfold_validation_accuracy_score':grid_search.best_score_, 
                                                       'validation_accuracy': validation_accuracy,
                                                       'precision_validation': Precision_validation,
                                                       'recall_validation': Recall_validation,
                                                       'F1_score_validation': F1_score_validation,
                                                       'train_accuracy': train_accuracy
                                                       }

def convert(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32): return float(o)  
        raise TypeError




def  evaluate_all_models(task_folder):

    model_params = {
        'logistic_regression': {
        'model': LogisticRegression,
        'params':{
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'penalty': ['none', 'l1', 'l2', 'elasticnet'],
        'C' : [100, 10, 1.0, 0.1, 0.01]
          }
        },

         'decision_tree':{
         'model':DecisionTreeClassifier,
         'params':{"splitter":["best","random"],
           "max_depth" : np.arange(1,13,3),
           "min_samples_leaf": np.arange(0.1, 1.0, 0.18),
           "min_weight_fraction_leaf":np.arange(0,0.5,0.1),
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
           }
         },
         'random_forest':{
         'model':RandomForestClassifier,
         'params' : {
         "max_features" : ["sqrt", "log2", None], # Number of features to consider at every split
         "max_depth" : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],# Maximum number of levels in tree
         "min_samples_split" : np.linspace(0.1, 1.0, 5, endpoint=True),# Minimum number of samples required to split a node
         "min_samples_leaf" : np.linspace(0.1, 1.0, 5, endpoint=True),# Minimum number of samples required at each leaf node
         "bootstrap": [True, False]# Method of selecting samples for training each tree
            }
         },
         'gradient_boosting' : {
         'model': GradientBoostingClassifier,
         'params': { 
         'learning_rate': [0.01,0.1,1],
           'n_estimators' : [1, 8 ,16, 32, 64, 100],
           'max_depth': [20, 30, 100, None],
           'min_samples_split' : np.linspace(0.1, 1.0, 5, endpoint=True),
           'min_samples_leaf' : np.linspace(0.1, 0.5, 4, endpoint=True),
           'max_features': ['auto', 'sqrt', 'log2']
                     }
         
                 }
           }
    
         
    # scores=[]
    for model_classes, mp in model_params.items():
        tuned_model, best_params, metrics =tune_classification_model_hyperparameters(model_class=mp['model'], X_train=X_train, y_train=y_train, 
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
    best_accuracy = -float('inf')
    best_f1 = float('inf')
    models_directory_list=os.listdir(models_directory)
    for model_name in models_directory_list:
        metrics_path = os.path.join(models_directory, model_name, 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)
            #val_r2 = metrics['validation_r2'] 
            validation_f1 = metrics['F1_score_validation']
            avg_kfold_validation_accuracy=metrics['avg-kfold_validation_accuracy_score']

            if avg_kfold_validation_accuracy > best_accuracy and validation_f1 > best_f1:
                best_model = model_name
                best_accuracy = avg_kfold_validation_accuracy
                best_f1 = validation_f1

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
    evaluate_all_models(task_folder = 'models/classification')#evaluate all model and return best one
      