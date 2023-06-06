# Modelling-Airbnb-s-property-listing-dataset-


> "Modelling Airbnb's Property Listing Dataset," is a comprehensive analysis and modeling project focused on Airbnb property listings. It encompasses various techniques and models to predict property prices 'Price_Night' and classify categorical variables such as 'Category', with multiple models trained and evaluated. The project consists of the following key components:

> Regression Models: The modelling.py file contains the implementation of various regression models to predict property prices labeled "Price_Night" using a set of input features.

>Classification Models: The classification.py file implements classification models to predict the categorical variable labeled "Category" in the CSV file clean_tabular_data.csv. 

>Neural Network Models: The neuralnetwork.py file showcases the implementation of a neural network model for property price prediction. Additionally, There is separate file named Neuralnetworkreuse.py, which contains the previous reused neural network framework. These neural network models are trained and evaluated on the dataset, leveraging deep learning techniques to improve prediction accuracy.

## Milestone 3

- Firstly, the tabular_data.py file contain functions which are designed to clean and process tabular data, specifically for the Airbnb property listing dataset. Below we will explain briefly what each function does in the processes.

- remove_rows_with_missing_ratings(data_frame): This function removes rows with missing values in specific columns of the DataFrame. It takes a pandas DataFrame as input and returns the DataFrame without the rows containing missing values.
  
```python
"""
def remove_rows_with_missing_ratings(data_frame):
    """_summary_: removes the rows with missing values in these columns.

    Parameters
    ----------
    data_frame : _type_: pandas dataframe 
        

    Returns
    -------
    _type_: pandas dataframe 
        _description_:returns dataframe without na values 
    """
    columns_with_na = ['Cleanliness_rating' , 'Accuracy_rating', 'Communication_rating', 'Location_rating',
                                'Check-in_rating', 'Value_rating']
    data_frame.dropna(axis=0, subset = columns_with_na, inplace=True)#axis 0 means drop row while axis=1 refers to columns
    data_frame.drop('Unnamed: 19', axis=1, inplace=True)#drop a column named 'Unnamed: 19' from the DataFrame
    return data_frame
"""
```

- combine_description_strings(data_frame): This function combines the list items into the same string in the "Description" column of the DataFrame. It removes any records with a missing description and removes the "About this space" prefix that every description starts with. It takes a pandas DataFrame as input and returns the modified DataFrame.

```python
"""def combine_description_strings(data_frame):
    """_summary_: combines the list items into the same string

    Parameters
    ----------
    data_frame : _type_: pandas dataframe
        _description_

    Returns
    -------
    _type_: dataframe
        _description_: The lists contain many empty quotes which should be removed. If you don't remove them before
    joining the list elements with a whitespace, they might cause the result to contain multiple whitespaces in places. 
    The function should take in the dataset as a pandas dataframe and return the same type. It should remove any records with a 
    missing description, and also remove the "About this space" prefix which every description starts with.
    """
    
    data_frame.dropna(axis=0, subset = 'Description', inplace=True)#drop row with column named description
    data_frame['Description'] = data_frame['Description'].astype(str)
    #data_frame["Description"] = data_frame["Description"].apply(lambda x: x.replace("'About this space', ", '').replace("'', ", '').replace('[', '').replace(']', '').replace('\\n', '. ').replace("''", '').split(" "))
    data_frame["Description"] = data_frame["Description"].apply(lambda x: x.replace('About this space', " "))
    
    data_frame['Description'].replace([r"\\n", "\n", r"\'"], [" "," ",""], regex=True, inplace=True)
    data_frame["Description"] = data_frame["Description"].apply(lambda x: " ".join(x))
    # apply(): It is a method in pandas that applies a function to each element of a column or DataFrame.
    # lambda x: " ".join(x): It is a lambda function that takes each element x of the "Description" column 
    # and joins the individual elements of x into a single string separated by a space.
    return data_frame"""
```
- set_default_feature_values(data_frame): This function fills empty values in the "guests", "beds", "bathrooms", and "bedrooms" columns with the number 1. It replaces the empty entries with 1 and converts the "bedrooms" and "guests" columns to integer type. It takes a pandas DataFrame as input and returns the modified DataFrame.

```python
"""def set_default_feature_values(data_frame):
    """_summary_:The "guests", "beds", "bathrooms", and "bedrooms" columns have empty values for some rows. 
    I Don't remove them, instead, I define a function called set_default_feature_values, and replace these entries with the number 1

    Parameters
    ----------
    data_frame : _type_: pandas dataframe
        _description_

    Returns
    -------
    _type_: dataframe
        _description_: A dataframe with the columns "guests", "beds", "bathrooms", and "bedrooms" filled with 1.
    """
      
    column_list = ["guests", "beds", "bathrooms", "bedrooms"]
    data_frame[column_list] = data_frame[column_list].fillna(1)
    data_frame=data_frame[data_frame['bedrooms'] != r"https://www.airbnb.co.uk/rooms/49009981?adults=1&category_tag=Tag%3A677&children=0&infants=0&search_mode=flex_destinations_search&check_in=2022-04-18&check_out=2022-04-25&previous_page_section_name=1000&federated_search_id=0b044c1c-8d17-4b03-bffb-5de13ff710bc"]# remove row with url link
    data_frame['bedrooms']=data_frame['bedrooms'].astype(int)
    data_frame['guests']=data_frame['guests'].astype(int)# convert guest and bedrooms from type object to type int 
    return data_frame"""
```
- clean_tabular_data(data_frame): This function sequentially calls the above three functions on the input DataFrame. It applies the cleaning operations step by step and returns the processed DataFrame.

```python
"""
def clean_tabular_data(data_frame):
    """_summary_:function called clean_tabular_data which takes in the raw dataframe, 
    calls these functions sequentially on the output of the previous one, and returns the processed data.
    

    Parameters
    ----------
    data_frame : _type_: pandas dataframe
        _description_

    Returns
    -------
    _type_: dataframe
        _description_: returns output of previous function 
    """
    data= remove_rows_with_missing_ratings(data_frame)
    data2=combine_description_strings(data)
    data3=set_default_feature_values(data2)
    return data3"""
```
- load_airbnb(df, label): This function takes a DataFrame (df) and a label (a single column) as input and returns the features and labels of the data in a tuple format.

```python
"""def load_airbnb(df, label):
    """_summary_: returns the features and labels of your data in a tuple like (features, labels).

    Parameters
    ----------
    df : _type_: dataframe 
        _description_
    label : _type_: one column of shape(829, 1)
        _description_
    """


    y=df[label]#target/label
    df.drop(label, axis=1, inplace=True)#features
       # text_columns_dataframe=df.select_dtypes('object')
    X= df.select_dtypes(exclude='object')#features
    return (X,y) #tuple of features and target respectively
"""
```
- save_clean_csv(): This function reads a the raw CSV file called 'listing.csv', applies the cleaning operations using the clean_tabular_data() function, and saves the cleaned DataFrame as a new CSV file called 'clean_tabular_data.csv'.

## Milestone 4: Regression models

- The modelling.py file loads tabular data, splits it into training, validation, and test sets, scales the data, trains regression models, evaluates their performance using various metrics, tunes their hyperparameters using the validation set, and saves the best model along with its hyperparameters and metrics.

- I started by importing the necessary libraries and modules for data loading, model selection, preprocessing, and evaluation.

- It loads the tabular data from a CSV file using the load_airbnb function from the tabular_data module. The target variable is set as the 'Price_Night' column.

- The data is split into training, validation, and test sets using the train_test_split function from sklearn.model_selection. The training set contains 70% of the data, while the validation and test sets each contain 15%.

- The feature data (X) is standardized using the StandardScaler from sklearn.preprocessing to ensure that all features have zero mean and unit variance.

- As a basline model A linear regression model, SGDRegressor from sklearn.linear_model, is trained on the standardized training data using stochastic gradient descent. The model is fit to the training data using the fit method with no tuned hyperparameters. The point of this task is to get a baseline to compare other more advanced models to try to improve upon.

- Predictions are made on the test set using the trained model, and the predictions are stored in y_test_pred and y_train_pred.  Various performance metrics are then computed using mean_squared_error, r2_score, and mean_absolute_error from sklearn.metrics. These metrics include the root mean squared error (RMSE) and R^2 score for both the training and test sets, which measure the model's accuracy and goodness of fit. These metrics will be compared later with the best model. The code is shown below:

```python
"""
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


np.random.seed(42)

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

#baseline model's metrics below
print("test RMSE (scikit-learn):", mean_squared_error(y_test, y_test_pred, squared=False))
print("test R2 (scikit-learn):", r2_score(y_test, y_test_pred))

print("train RMSE (scikit-learn):", mean_squared_error(y_train, y_train_pred, squared=False))
print("train R2 (scikit-learn):", r2_score(y_train, y_train_pred))"""
```
- For a better understanding about what is going on under the hood, I created a custom function called custom_tune_regression_model_hyperparameters which performs a grid search over the specified hyperparameters to find the best combination. It creates model instances with different hyperparameter combinations, trains them on the training data, and evaluates their performance on the validation set. The best model with the lowest validation RMSE is selected and retrained on the training data. Finally, the performance metrics are computed on the test and train sets. The function is show

```python
"""
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
   
   
    return best_model, best_hyperparams, metrics"""
```
- During Tuning the tune_regression_model_hyperparameters was instead used. This also tunes hyperparameters using grid search but utilizes the GridSearchCV function from sklearn.model_selection. It fits the model to the training data and performs a grid search over the specified parameter grid testing out all possible combinations of hyperparameters for ech model. The best model with the lowest validation RMSE is selected and retrained on the training data. The performance metrics (R2 score and root mean square error) are then computed on all 3 sets.

- I used cross validation with cv=5 parameter within GridsearchCV()  to assess how well the model generalizes to unseen data and reduces the impact of any particular subset of the data on the evaluation. The dataset is split into 5 equal parts, or folds. The grid search algorithm will then perform training and evaluation on each combination of hyperparameters using 5-fold cross-validation.

- During each iteration of cross-validation, one fold is used as the validation set, while the remaining folds are used for training the model. This process is repeated 5 times, with each fold being used as the validation set exactly once. The performance metrics for each combination of hyperparameters are averaged across the 5 folds to determine the overall performance.

```python
"""

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
        
        y_test_pred = best_model.predict(X_test)
        train_R2=best_model.score(X_train, y_train)
        validation_mse =  mean_squared_error(y_validation, y_validation_pred)
        validation_rmse = validation_mse**(1/2)
        validation_R2=best_model.score(X_validation, y_validation)
        train_R2=best_model.score(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_R2 = best_model.score(X_test, y_test)
        #cv_score=cross_val_score(best_model, X_train, y_train, cv = 10)
        #mae = mean_absolute_error(y_test, y_pred)# mean absolute error
        metrics = {'avg-kflod_validation_rmse' :grid_search.best_score_, 
                                                       'validation_rmse': validation_rmse,
                                                       'train_rsquared': train_R2,
                                                       'validation_rsquared': validation_R2,
                                                       'train_rmse': train_rmse,
                                                       'test_rmse' : test_rmse,
                                                       'test_R2': test_R2
                                                       }

        return  best_model, grid_search.best_params_, metrics
"""
```
- The convert function is defined to handle the conversion of NumPy integer types to floats when saving models using json.dump.

- The save_model function saves the best model, hyperparameters, and performance metrics in the specified folder. It uses json.dump and joblib.dump from json and joblib modules, respectively. These are shown below.

```python
"""

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
            
"""
```
- evaluate_all_models function evaluates and tunes the hyperparameters for multiple regression models. It iterates over a dictionary of model parameters, tunes each model, and saves the best model, hyperparameters, and metrics in separate files within the specified task folder.

```python
"""

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
           "max_leaf_nodes":[None,10,50,60,70,90],
           "ccp_alpha": [0.01, 0.1, 1, 10]
           }
         },
         'random_forest':{
         'model':RandomForestRegressor,
         'params' : {
         "max_features" : ["sqrt", "log2", None], # Number of features to consider at every split
         "max_depth" : [5, 10, 30, 60, 100, None],# Maximum number of levels in tree
         "min_samples_split" : np.linspace(0.1, 1.0, 5, endpoint=True),# Minimum number of samples required to split a node
         "min_samples_leaf" : np.linspace(0.1, 1.0, 5, endpoint=True),# Minimum number of samples required at each leaf node
         "bootstrap": [True, False],# Method of selecting samples for training each tree
         "ccp_alpha": [0.01, 0.1, 1,10] 
          }
         },
         'gradient_boosting' : {
         'model': GradientBoostingRegressor,
         'params': { 
         'learning_rate': [0.01,0.1,1],
           'n_estimators' : [1, 8 ,16, 32, 64, 100],
           'max_depth': [5, 10, 30, 60, 100, None],
           'min_samples_split' : np.linspace(0.1, 1.0, 5, endpoint=True),
           'min_samples_leaf' : np.linspace(0.1, 1.0, 5, endpoint=True),
           'max_features': ['auto', 'sqrt', 'log2'],
           'min_impurity_decrease': [0.01, 0.1, 1, 10]
         
                 }
        },
        'sgdrregressor': {
         'model': SGDRegressor,
         'params':{
         'alpha': 10.0 ** -np.arange(1, 7),
        'loss': ['huber', 'epsilon_insensitive', 'squared_error'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'max_iter': [1000,10000]
            
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
"""
```
- find_best_model function is responsible for finding the best regression model among the saved models in the specified results folder. It loads each model, retrieves the associated metrics, and compares the average kfold rmse on the validation set to determine the best model based on a chosen evaluation metric. The function returns the best model's hyperparameters and metrics.
```python
"""
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
        avg_kfold_val_rmse = metrics["avg-kflod_validation_rmse"]*-1
        validation_r2=metrics['validation_rsquared']

        if avg_kfold_val_rmse < best_rmse: #and validation_r2 > best_r2:
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

    with open(os.path.join('models/best_regression_model','hyperparameters.json') , mode='w') as f:
         json.dump(optimum_params, f)
    
    with open(os.path.join('models/best_regression_model','metrics.json' ) , mode='w') as f:
         json.dump(optimum_metrics, f)

    return optimum_metrics, optimum_params, optimum_model

"""
```
- The  visualize_graphs function is used to visualize graphs related to the regression models. It loads the saved models from the specified model folder and generates visualizations such as learning curves, residual plots, to analyze and interpret the models' performance. 


```python
"""
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
                
"""
```
## Milestone 5: Classification models

## Over-fitting

## Milestone 6: Neural network

## Best parameterised network

## Milestone 7- Reusing the framework

## Conclusion 

- Continue this process for every milestone, making sure to display clear understanding of each task and the concepts behind them as well as understanding of the technologies used.

- Also don't forget to include code snippets and screenshots of the system you are building, it gives proof as well as it being an easy way to evidence your experience!

## Conclusions

- Maybe write a conclusion to the project, what you understood about it and also how you would improve it or take it further.

- Read through your documentation, do you understand everything you've written? Is everything clear and cohesive?