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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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



# this will print the metrics for both the train and test sets of the baseline model which we will compare the the tuned one.
print("Accuracy_test:", accuracy_score(y_test, y_pred_test))
print("Precision_test:", precision_score(y_test, y_pred_test, average="macro"))
print("Recall_test:", recall_score(y_test, y_pred_test, average="macro"))
print("F1 score_test:", f1_score(y_test,y_pred_test, average="macro"))


print("Accuracy_train:", accuracy_score(y_train, y_pred_train))
print("Precision_train:", precision_score(y_train, y_pred_train, average="macro"))
print("Recall_train:", recall_score(y_train, y_pred_train, average="macro"))
print("F1 score_train:", f1_score(y_train, y_pred_train, average="macro"))


def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_validation ,y_validation, X_test, y_test, parameter_grid):
        """_summary_: Tune hyperparameters for a classification model.

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
        y_test_pred= best_model.predict(X_test)
        y_validation_pred = best_model.predict(X_validation)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        validation_accuracy = accuracy_score(y_validation, y_validation_pred)

        Precision_validation= precision_score(y_validation, y_validation_pred, average="macro")
        Precision_train= precision_score(y_train, y_train_pred, average="macro")
        Precision_test = precision_score(y_test, y_test_pred, average="macro")
        
        
        Recall_validation= recall_score(y_validation, y_validation_pred, average="macro")
        Recall_test= recall_score(y_test, y_test_pred, average="macro")
        Recall_train= recall_score(y_train, y_train_pred, average="macro")
    
        F1_score_validation = f1_score(y_validation, y_validation_pred, average="macro")
        F1_score_test = f1_score(y_test, y_test_pred, average="macro")
        F1_score_train = f1_score(y_train, y_train_pred, average="macro")
        metrics = {'avg-kfold_validation_accuracy_score':grid_search.best_score_, 
                                                       'validation_accuracy': validation_accuracy,
                                                       'test_accuracy': test_accuracy,
                                                       'train_accuracy': train_accuracy,
                                                       'precision_validation': Precision_validation,
                                                       'precision_train': Precision_train,
                                                       'precision_test': Precision_test,
                                                       'recall_validation': Recall_validation,
                                                       'recall_test': Recall_test,
                                                       'recall_train': Recall_train,
                                                       'F1_score_validation': F1_score_validation,
                                                       'F1_score_train': F1_score_train,
                                                       'F1_score_test': F1_score_test,
                                                       }


        return  best_model, grid_search.best_params_, metrics

def convert(o):
     #this function is used to convert specific NumPy integer types to float, and it raises an error for other types. 
        
    if isinstance(o, np.int64) or isinstance(o, np.int32): return float(o)  
    raise TypeError





def  evaluate_all_models(task_folder):
    """_summary_ : Evaluate and tune hyperparameters for multiple regression models. The loop iterated over the dict and saves each tuned model

    Parameters
    ----------
    task_folder : _type_: str
        _description_, by default 'models/regression' which is the derectory the models will be saved
    """

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
           "max_depth" : [5,6,7],
           "min_samples_leaf": np.linspace(0.01,0.50,4),
           "min_weight_fraction_leaf":np.arange(0.1, 0.3,0.05),
           "max_features":["auto","log2","sqrt"],
           "max_leaf_nodes":[25, 30, 40],
           "ccp_alpha": [0.01, 0.5,  0.1]
           }
         },
         'random_forest':{
         'model':RandomForestClassifier,
         'params' : {
         "max_features" : ["sqrt", "log2", "auto"], # Number of features to consider at every split
         "max_depth" : [84, 87, 86],# Maximum number of levels in tree
         "min_samples_split" : np.linspace(0.01,0.50,4),# Minimum number of samples required to split a node
         "min_samples_leaf" : np.arange(0.08, 0.1, 0.005),# Minimum number of samples required at each leaf node
         "bootstrap": [True, False],
         "ccp_alpha": [0.01, 0.03, 0.05]# Method of selecting samples for training each tree
            }
         },
         'gradient_boosting' : {
         'model': GradientBoostingClassifier,
         'params': { 
         'learning_rate': [0.4, 0.5, 0.8],
           'n_estimators' : [100, 1000, 10000],
           'max_depth': [30, 40, 50],
           'min_samples_split' : np.linspace(0.01,0.4, 4),
           'min_samples_leaf' : [0.01, 0.5 ,0.1],
           'max_features': ['auto', 'sqrt', 'log2'],
           'min_impurity_decrease': [0.01, 0.03, 0.05]
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
    """_summary_:  summary_
    this fucntion takes the models_directory as input, which is the directory where the saved models are stored.
    It initializes variables to keep track of the best model, best validation accuracy, and best validation f1 score. 
    Initially both the best f1 and accuracy  is set to negative infinity, 
    It iterates through each model in the models_directory and reads the metrics from the corresponding metrics file (metrics.json).
    It compares the average k-fold validation accuracy and validation f1 score of the current model with the
    best accuracy and best f1 score values obtained so far. 
    If the current model has a greater accuracy and a higher f1score, it becomes the new best model.
    After iterating through all the models, it identifies the path of the best model's metrics file, hyperparameters file, and model file.
    It loads the metrics, hyperparameters, and model using the identified paths.
    Finally, it returns the optimum metrics, optimum hyperparameters, and optimum model.

    Parameters
    ----------
    models_directory : _type_
        _description_
    """
    
    best_model = None
    best_accuracy = -float('inf')
    best_f1 = -float('inf')
    models_directory_list=os.listdir(models_directory)
    for model_name in models_directory_list:
        metrics_path = os.path.join(models_directory, model_name, 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)
            #val_r2 = metrics['validation_r2'] 
        validation_f1 = metrics['F1_score_validation']
        avg_kfold_validation_accuracy=metrics['avg-kfold_validation_accuracy_score']

        if avg_kfold_validation_accuracy > best_accuracy:
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

    
    with open(os.path.join('models/best_classification_model','model.joblib' ) , mode='wb') as f:
         joblib.dump(optimum_model, f)
    
    with open(os.path.join('models/best_classification_model','hyperparameters.json' ) , mode='w') as f:
         json.dump(optimum_params, f)
    
    with open(os.path.join('models/best_classification_model','metrics.json' ) , mode='w') as f:
         json.dump(optimum_metrics, f)



    return optimum_metrics, optimum_params, optimum_model
    #         maxvalr2.append(validation_r2)
    # #print(min(minvalrmse))
    # model_to_load=models_directory_list[np.argmax(maxvalr2)]
    # model_path = os.path.join(models_directory, model_to_load, 'metrics.json')
    #     with open(metrics_path) as f:

# def visualise_confusion_matrix(folder, X_test, y_test):
#     """_summary_: provides graphics for the normalised and unormalised confusion matrix

#     Parameters
#     ----------
#     folder : _type_: str
#         _description_: the directory to find the best model
#     X_test : _type_:numpy.ndarray
#             _description_: The normalized test set for the features
#     y_test : _type_:numpy.ndarray
#             _description_: The normalized test set for the label
#     """
#     best_model_path = os.path.join(folder, 'model.joblib')
#     with open(best_model_path, mode='rb') as f:
#         best_model = joblib.load(f)
    
#     y_pred = best_model.predict(X_test)
#     cm = confusion_matrix(y_test, y_pred)
#     normalized_cm = cm / cm.sum()  # Compute the normalized confusion matrix
    
#     # Plotting the confusion matrix
#     fig, ax = plt.subplots(figsize=(8, 6))
#     display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
#     display.plot(ax=ax, cmap='Blues')
#     ax.set_title('Confusion Matrix')
#     plt.show()
    
#     # Plotting the normalized confusion matrix
#     fig, ax = plt.subplots(figsize=(8, 6))
#     display_normalized = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=np.unique(y_test))
#     display_normalized.plot(ax=ax, cmap='Blues')
#     ax.set_title('Normalized Confusion Matrix')
#     plt.show()


def visualise_confusion_matrix(models_directory, X_test, y_test):
    model_files = os.listdir(models_directory)

    # Create a figure for the normalized confusion matrix
    fig_norm_cm, axes_norm_cm = plt.subplots(2, 2, figsize=(25, 8))
    fig_norm_cm.suptitle('Normalized Confusion Matrix')

    # Create a figure for the confusion matrix
    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(15, 8))
    fig_cm.suptitle('Confusion Matrix')

    for i, model_name in enumerate(model_files):
        metrics_path = os.path.join(models_directory, model_name, 'model.joblib')
        with open(metrics_path, 'rb') as f:
            model = joblib.load(f)

        y_pred = model.predict(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix
        normalized_cm = cm / cm.sum()

        # Plot normalized confusion matrix
        row = i // 2
        col = i % 2
        ax_norm_cm = axes_norm_cm[row, col]
        display_norm_cm = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=np.unique(y_test))
        display_norm_cm.plot(ax=ax_norm_cm, cmap='Blues')
        ax_norm_cm.set(title=f'Normalized Confusion Matrix ({model_name})')
        
        # Adjust font size of the numbers on the normalized confusion matrix plot
        for text in display_norm_cm.text_.ravel():
            text.set_fontsize(8)  # Set font size of the numbers
        plt.setp(ax_norm_cm.get_xticklabels(), rotation=25, ha='right')
    

        # Plot confusion matrix
        ax_cm = axes_cm[row, col]
        display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
        display_cm.plot(ax=ax_cm, cmap='Blues')
        ax_cm.set(title=f'Confusion Matrix ({model_name})')
        plt.setp(ax_cm.get_xticklabels(), rotation=25, ha='right')
        
    # Adjust spacing between subplots in the normalized confusion matrix figure
    #fig_norm_cm.tight_layout(pad=6.5)
    fig_norm_cm.tight_layout()
    # Adjust spacing between subplots in the confusion matrix figure
  #  fig_cm.tight_layout(pad=2.5)
    # Show the plots
    
    # Adjust spacing between subplots in the normalized confusion matrix figure
    fig_norm_cm.subplots_adjust(right = 0.9,    # the right side of the subplots of the figure
      bottom = 0.1 ,  # the bottom of the subplots of the figure
      top = 0.9   ,   # the top of the subplots of the figure
      wspace = 0.01  , # the amount of width reserved for blank space between subplots
      hspace = 0.8   )
    # Adjust spacing between subplots in the confusion matrix figure
    fig_cm.subplots_adjust( left  = 0.125,  # the left side of the subplots of the figure
      right = 0.9,    # the right side of the subplots of the figure
      bottom = 0.1 ,  # the bottom of the subplots of the figure
      top = 0.9   ,   # the top of the subplots of the figure
      wspace = 0.2  , # the amount of width reserved for blank space between subplots
      hspace = 0.8   )
    plt.show()

if __name__ == "__main__" :
    # random seed at the begining ensure constant output
    #
   #cd modeevaluate_all_models('models/classification')
     
   print(find_best_model('models/classification'))
    
  #   tuned_model, best_params, metrics =tune_regression_model_hyperparameters(model_class=SGDRegressor, X_train=X_train, y_train=y_train, 
                     #                            X_validation=X_validation,y_validation=y_validation,X_test= X_test,y_test= y_test, 
                  #                                parameter_grid=hyperparameter_grid)


    #  save_model("models/regression")

    #visualise_confusion_matrix('models/classification', X_test, y_test )
    