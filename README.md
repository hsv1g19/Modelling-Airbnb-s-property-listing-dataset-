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
- load_airbnb(df, label): This function takes a DataFrame (df) and a label (a single column) as input and returns the features and labels of the data in a tuple format. This function can them be used to set the features and labels in other scrips.

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

- Stochastic Gradient Descent is an iterative optimization algorithm that aims to find the optimal values for the regression coefficients by minimizing the mean squared error (MSE) or other loss functions. It performs updates to the coefficients after each training example, rather than after processing the entire training set as in batch gradient descent.

- During training, SGDRegressor updates the coefficients by considering a random sample (or a minibatch) of training examples at each iteration. It computes the gradient of the loss function with respect to the coefficients using this subset of data and performs a small update to the coefficients in the direction that minimizes the loss.


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
- evaluate_all_models function evaluates and tunes the hyperparameters for multiple regression models including Decision tree, Random forest, Gradient boosting and SGDR. It iterates over a dictionary of model parameters, tunes each model, and saves the best model, hyperparameters, and metrics in separate files within the specified task folder.

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
- find_best_model function is responsible for finding the best regression model among the saved models in the specified results folder. It loads each model, retrieves the associated metrics, and compares the average kfold rmse on the validation set to determine the best model. The function returns the best model's hyperparameters and metrics and also saves each of them in a directory called 'models/best_regression_model'.

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
- The  visualize_graphs function is used to visualize graphs related to the regression models. It loads the saved models from the specified model folder and generates visualizations which allow comparison of the true vs predicted values and the residuals for different regression models, providing insights into how well each model is performing in terms of predicting the target variable.

- It takes three inputs: models_directory, which is the directory where the trained models are stored, X_test, which is the input data for testing, and y_test, which is the corresponding true labels for the test data.

- It retrieves the list of model files present in the models_directory.

- It creates two figures: fig_res for the residual plots and fig_other for other plots such as true vs predicted values.

- It iterates over each model file in the model_files list and for each model, it loads the trained model using joblib.load().

- It makes predictions on the test data using the loaded model (model.predict(X_test)), obtaining the predicted labels y_pred.

- It calculates the residuals by subtracting the predicted labels from the true labels (y_test - y_pred).

- It plots the true values vs predicted values on the corresponding subplot in the fig_other figure using ax_other.scatter(). 

- It plots the residuals on the corresponding subplot in the fig_res figure using ax_res.scatter(). It also adds a horizontal line at y=0 using ax_res.axhline() to indicate the zero residual line. 


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

## Results

![Alt text](<project - Copy/truevspred.png>)
| model | SGDR  | gradient boosting| random forest|decision tree|
| ----- | ------| --------- |  -------| -------|
| avg-kflod_validation_rmse| ---------| --------- |  -------|-------|
| validation_rmse | ---------|           | -------|-------|
| train_rsquared | ---------|           | -------|-------|
| validation_rsquared| ---------|           | -------|-------|
| train_rmse  | Content  |           |-------|-------|
| test_rmse  | Content  |           |-------|-------|
| test_rmse  | Content  |           |-------|-------|
| test_rmse  | Content  |           |-------|-------|


![Alt text](<project - Copy/residuleplot.png>)

## Milestone 5: Classification models
- Using the same framework to the regression models in modelling.py, I imported my load_airbnb function from the tabular_data module as well as libraries such as pandas, numpy, matplotlib, sklearn. I then load in the clean_data_frame is loaded from the CSV file called 'clean_tabular_data.csv', the load_airbnb function is called to extract features (X) and labels (y) from the data, in this case my label was the 'Category' column which includes 'Treehouse', 'Chalets', 'Amaizing pools', 'Offbeat' and 'Beachfront.

- I then split the data as before where the training set contains 70% of the data, while the validation and test sets each contain 15%. And again scale the data using StandardScaler

- As a basline model A LogisticRegression model from scikit-learn was used.linear_model, is trained on the standardized training data.

- Logistic regression is used for binary classification problems. It uses the logistic function, also known as the sigmoid function, to model the relationship between the independent variables (features) and the binary dependent variable (the target variable).

- The logistic regression method estimates the probability that a given instance belongs to a particular class. It models the relationship between the input variables and the probability of the target variable taking on a specific value. It is givern by:

- sigmoid(x) = 1 / (1 + exp(-x))

- In logistic regression, the input features are linearly combined with their corresponding weights, and the result is passed through the logistic function to obtain the predicted probability. The weights are estimated during the training process using optimization algorithms such as maximum likelihood estimation or gradient descent.

- The predicted probability can then be used to make binary predictions by applying a threshold. For example, if the predicted probability is greater than 0.5, the instance is classified as belonging to one class, and if it is less than or equal to 0.5, it is classified as belonging to the other class.

- The model is fit to the training data using the fit method with no tuned hyperparameters. The point of this task is to get a baseline to compare other more advanced models to try to improve upon.

- Predictions are made on the test set using the trained model, and the predictions are stored in y_test_pred and y_train_pred. Various performance metrics are then computed using accuracy score, precision score, recall_score, from sklearn.metrics. These metrics include the f1_score score for both the training and test sets, which measure the model's accuracy and goodness of fit. These metrics will be compared later with the best model. The code is shown below:

```python
"""
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

"""
```
- For tuning each classification model I used an identical tune function as the regression case called tune_classification_model_hyperparameters which used GridSearchCV' and tests out all possible combinations of hyperparameters for ech model using the parameter grid provided as an argument,and used cross validation with cv=5 . The best model with the highest accuracy score is then selected (the accuracy score measures the proportion of correctly predicted labels to the total number of samples) and retrained on the training data. The performance metrics (accuracy_score,  precision_score, recall_score and f1_score) are then computed on all 3 sets.

```python
"""

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


        return  best_model, grid_search.best_params_ metrics"""
```
- The convert function is defined to handle the conversion of NumPy integer types to floats when saving models using json.dump.

- The save_model function is imported from modelling.py which is called in the loop which iterates over each model in the model_params dictionary.within the evaluate_all_models function, with the folder path 'models/classificiation' and saves the best model, hyperparameters, and performance metrics in the specified folder. It uses json.dump and joblib.dump from json and joblib modules, respectively. These are shown below.

- evaluate_all_models function evaluates and tunes the hyperparameters for multiple classification models including Decision tree, Random forest, Gradient boosting and SGDR. It iterates over a dictionary of model parameters, tunes each model, and saves the best model, hyperparameters, and metrics in separate files within the specified task folder.
```python
"""
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
"""
```
- find_best_model function is responsible for finding the best classification model among the saved models in the specified results folder. It loads each model, retrieves the associated metrics, and compares the average kfold validation accuracy  to determine the best model based. The function returns the best model's hyperparameters and metrics and also saves each of them in a directory called 'models/best_classification_model'.

```python
"""
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



    return optimum_metrics, optimum_params, optimum_model"""
```
- The function visualise_confusion_matrix is used to generate and display confusion matrices and normalized confusion matrices for multiple models. The function visualizes the performance of multiple models by showing their confusion matrices and normalized confusion matrices, allowing for a comparison of how well each model is predicting the different classes.:

- It takes three inputs: models_directory, which is the directory where the trained models are stored, X_test, which is the input data for testing, and y_test, which is the corresponding true labels for the test data.

- It retrieves the list of model files present in the models_directory.

- It creates two figures: fig_norm_cm for the normalized confusion matrices and fig_cm for the confusion matrices. These figures have subplots arranged in a 2x2 grid.

- It iterates over each model file in the model_files list.

- For each model, it loads the trained model using joblib.load().

- It makes predictions on the test data using the loaded model (model.predict(X_test)), obtaining the predicted labels y_pred.

- It computes the confusion matrix (cm) by comparing the true labels (y_test) with the predicted labels (y_pred).

- It normalizes the confusion matrix by dividing each element by the sum of all elements, obtaining normalized_cm.

- It plots the normalized confusion matrix on the corresponding subplot in the fig_norm_cm figure using ConfusionMatrixDisplay.plot(). 

- It plots the confusion matrix on the corresponding subplot in the fig_cm figure using ConfusionMatrixDisplay.plot(). 

## Results
![Alt text](<project - Copy/confusionmatrix.png>)

![Alt text](<project - Copy/normalizedconfusionmatrix.png>)
## Milestone 6: Neural network

The first task in creating a configurable neural network was to Create a PyTorch Dataset called AirbnbNightlyPriceRegressionDataset that returns a tuple of (features, label) when indexed. The features are a tensor of the numerical tabular features of the house. The second element is a scalar of the price per night. I then 
create a dataloader for the train set and test set that shuffles the data and further split the train set into train and validation.

To start with I import the neccessary modules here are a brief description of each:

- import json: This module provides functions for working with JSON data. It is used for handling JSON files or data structures.

- import os: This module provides a way to interact with the operating system. It is used for various operations such as file and directory manipulation, path handling, etc.

- import time: This module provides functions for working with time-related operations. It is used to measure the execution time of certain code sections or to introduce delays in the program.

- from torch.utils.data import Dataset, DataLoader, random_split: These classes and functions are part of the torch.utils.data module, which provides tools for working with data loading and processing in PyTorch.

- Dataset is an abstract class representing a dataset. It is used as a base class for creating custom datasets.
DataLoader is a class that provides an iterable over a dataset. It is used to efficiently load data in batches during training or evaluation.
random_split is a function that is used to randomly split a dataset into non-overlapping new datasets. It is commonly used to create train/validation/test splits.

- import torch: This is the main package for PyTorch. It provides functionalities for tensor operations, neural networks, optimization algorithms, etc.

- from torchmetrics import R2Score: This is a class from the torchmetrics package, which provides various metrics for evaluating machine learning models. R2Score is a metric for measuring the coefficient of determination (R-squared) for regression problems.

- import torch.nn as nn: This module provides classes for defining and working with neural networks in PyTorch. It includes various types of layers, loss functions, and activation functions.

- import torch.nn.functional as F: This module provides functional versions of the neural network layers and operations. It is often used in conjunction with the nn module for defining custom neural network architectures.

- from torch.utils.tensorboard import SummaryWriter: This class is used for writing TensorBoard logs. TensorBoard is a visualization toolkit for PyTorch that allows tracking and visualizing various metrics, model graphs, and other information during training.

- import yaml: This library provides functions for working with YAML files. YAML is a human-readable data serialization format.

- import datetime: This module provides classes for working with dates and times. It is used to handle timestamps and time-related operations.

- import itertools: This module provides various functions for creating iterators and combining iterable objects. It is used for generating combinations or permutations of elements from a given set.

- I then create an AirbnbNightlyPriceImageDataset class:

- This class represents the dataset used in the neural network model. It inherits from the torch.utils.data.Dataset class.

- The __init__ method initializes the dataset by loading a CSV file named clean_tabular_data.csv. It then calls the load_airbnb function from the tabular_data module to extract the necessary features and labels.
- The __getitem__ method is used to retrieve a specific sample from the dataset. It takes an index idx as input and returns a tuple of features and labels for that index. The features and labels are converted to PyTorch tensors using torch.tensor and cast to the float data type.
- The __len__ method returns the total number of samples in the dataset.

- get_random_split function: This function performs a random split on the given dataset to create training, validation, and test sets.
The function takes the dataset as input and uses the random_split function from torch.utils.data to split the dataset. It assigns 70% of the data to the train_set and the remaining 30% is split equally between the validation_set and test_set (15% each).
The function returns a tuple containing the three subsets: train_set, validation_set, and test_set.

- get_data_loader function:
This function performs a random split on the dataset and creates data loaders for the train, validation, and test sets.
The function takes the dataset as input and an optional batch_size parameter (default: 32) to determine the batch size for the data loaders.
It calls the get_random_split function to obtain the train, validation, and test sets.
Then, it uses the DataLoader class from torch.utils.data to create data loaders for each set. The data loaders allow for efficient batch processing during training and evaluation. The batch_size parameter determines the number of samples per batch, and the shuffle parameter is set to True to shuffle the data at each epoch.
The function returns a dictionary (data_loader) containing the data loaders for the train, validation, and test sets, accessible by the keys "train", "validation", and "test" respectively.

- These functions are designed to prepare the dataset and create data loaders for training, validation, and testing the neural network model. The code is shown below:

```python
""" import json
import os
import time
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
"""
```
- In task 2 I defined a PyTorch model class containing the architecture for a fully connected neural network which can perform a forward pass on a batch of data and produce an output of the correct shape, the code is shown below.

- class NeuralNetwork(nn.Module): This class represents a neural network model. It initializes the model's layers based on the provided configuration and defines the forward pass. The configuration includes the width and depth of hidden layers, and the input and output dimensions of the model.

- def forward(self, features): This method implements the forward pass of the neural network model. It takes input features and passes them through the model's layers to produce an output.

- def train(model, data_loader, optimizer, epochs=15): This function trains a neural network model. It takes the model, data loaders for train, validation, and test sets, and an optimizer as inputs. It performs the training loop for the specified number of epochs. Within each epoch, it iterates over batches of data, computes predictions, calculates loss, updates model parameters, and records performance metrics. It returns a dictionary containing the performance metrics of the trained model.

- Overall, these functions are responsible for defining and training a neural network model using PyTorch, including the model architecture, forward pass, and training loop.

```python
"""class NeuralNetwork(nn.Module):
    
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
"""
```
- The next task was to create a configuration file to change the characteristics of the model, the code is shown below.

- def generate_nn_configs(): This function generates configurations for neural network models. It returns a list of dictionaries representing different configurations. Each dictionary contains keys such as 'optimiser' (name of the optimizer), 'params' (optimizer parameters), 'hidden_layer_width' (width of hidden layers), and 'depth' (number of hidden layers).

- def convert_all_params_to_yaml(all_params, yaml_file): This function converts the generated configurations to a YAML file. It takes the list of configurations (all_params) and the path to the YAML file (yaml_file) as inputs. It writes the configurations to the YAML file using the yaml.safe_dump() function.

- def get_nn_config(yaml_file): This function retrieves the neural network configurations from a YAML file. It takes the path to the YAML file (yaml_file) as input. It reads the configurations from the YAML file using the yaml.safe_load() function and returns them as a list of dictionaries.

  
```python
"""
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


"""
```

- In order to save my models I Adaptws the earlier function called save_model so that it detects whether the model is a PyTorch module, and if so, saves the torch model in a file called model.pt, its hyperparameters in a file called hyperparameters.json, and its performance metrics in a file called metrics.json. These are all saved in a new folder called neuralnetwroks and within it a folder called regression.

- The metrics includes:

- The RMSE loss of the model under a key called RMSE_loss for training, validation, and test sets
- The R^2 score of the model under a key called R_squared for training, validation, and test sets
- The time taken to train the model under a key called training_duration
- The average time taken to make a prediction under a key called inference_latency
Every time a model is trained a new folder is created whose name is the current date and time.

- The function first checks if the model is an instance of torch.nn.Module. If it is not, a message is printed indicating that the model is not a PyTorch Module.

If the model is a PyTorch Module, the function proceeds to create the necessary directories in the folder path and generate a unique timestamp for the model folder. It then saves the model's state dictionary to a file called model.pt in the model folder.

The optimizer parameters are saved in a JSON file called hyperparameters.json, and the performance metrics are saved in a JSON file called metrics.json, both located in the model folder.


  
```python
"""
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
            json.dump(performance_metrics, fp)  """
```
- The next task was to get optimiser instances of all configurations. This is shown in the following code snippet below.

- def get_optimiser(yaml_file, model): This function is used to create optimizer instances based on the configurations specified in a YAML file. The function takes the following parameters:

- yaml_file: The path to the YAML configuration file.
- model: The model for which optimizers will be created.
The function first generates neural network configurations using the generate_nn_configs() function and saves them to the specified yaml_file using the convert_all_params_to_yaml() function. It then reads the configurations from the YAML file using the get_nn_config() function.

- Next, the function iterates over each optimizer configuration in the config list. It extracts the optimizer type and parameters from each configuration. The supported optimizer types are 'Adadelta', 'SGD', 'Adam', and 'Adagrad'. If an unknown optimizer type is encountered, a ValueError is raised.

- For each optimizer configuration, an optimizer instance is created based on the optimizer type and model's parameters. The optimizer parameters are passed to the optimizer constructor using the **optimiser_params syntax. The created optimizer instances are stored in a list.

- Finally, the function returns the list of optimizer instances.
  
```python
"""
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
    
    return optimisers"""
```
- The last task involved finding the best parameterised network through a function called find_best_nn. This function trains the neural network model with different optimizers based on the configurations specified in the YAML file, evaluates the models based on performance metrics, saves the best model's hyperparameters and metrics, and returns the best model's name, hyperparameters, and metrics.

- def find_best_nn(yaml_file, model, folder): This function is used to find the best neural network model based on performance metrics. It takes the following parameters:

- yaml_file: The path to the YAML configuration file.
model: The neural network model.
- folder: The folder path where the model will be saved.
The function starts by obtaining a list of optimizer instances by calling the get_optimiser() function with the yaml_file and model as arguments.

- Next, the function iterates over each optimizer instance in the optimisers list. It prepares the dataset and data loader, trains the model using the train() function with the given optimizer, and retrieves the performance metrics. The optimizer's hyperparameters are extracted and saved using the save_model() function.

- The function then scans the directory models/neural_networks/regression for subdirectories and retrieves the paths of the metrics.json files within those subdirectories. It keeps track of the best root mean square error (RMSE) loss encountered so far.

- For each metrics.json file, the function reads the file and retrieves the validation RMSE loss. If the current validation RMSE loss is better than the previous best RMSE loss, the best RMSE loss is updated, and the name of the best model is recorded.

- After evaluating all the metrics.json files, the function determines the directory path of the best model based on the best model name. It creates a new directory best_model in the models/neural_networks/regression directory and constructs paths for the best model's metrics, hyperparameters, and model file.

- The hyperparameters and metrics of the best model are loaded from the corresponding files in the best model's directory. The hyperparameters and metrics are then saved in the best_model directory.

- Finally, the function loads the model state dictionary from the best model's directory and saves it as model.pt in the best_model directory. The function returns a tuple containing the best model name, its hyperparameters, and performance metrics.

```python
"""
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

    return best_name, params, metrics    """
```
  
## Results
- Below is the neuralnetwork diagram for my model. It shows 11 input features, 5 hidden layers with a depth of 5 and fiving 1 output layer.within the neuralnetwork class I have used the ReLU function via nn.ReLU() from pytorch.

- The purpose of an activation function in a neural network is to introduce non-linearity into the model. Without non-linear activation functions, a neural network would be equivalent to a linear regression model, as the composition of linear operations is still a linear operation. By applying non-linear activation functions, neural networks can learn complex and non-linear relationships between input features and output predictions.

- ReLU is a popular choice for an activation function due to its simplicity and effectiveness. It returns the input value if it is positive or zero, and otherwise, it returns zero. ReLU introduces non-linearity by allowing the model to activate or deactivate certain neurons based on their input values. This can help the model learn more complex patterns and make the network more expressive.In order to train the model we use something called backpropogation.

- Backpropagation is an algorithm used to train neural networks by computing the gradients of the loss function with respect to the parameters of the network. It allows the network to learn from the training data and update its weights and biases to minimize the loss function.

- Here's a high-level overview of how backpropagation works:

- Forward Pass: During the forward pass, the input data is passed through the layers of the neural network, and the output prediction is computed. Each layer performs a weighted sum of the inputs, applies an activation function (such as ReLU), and passes the output to the next layer.

- Loss Calculation: The output of the neural network is compared with the true labels of the training data, and a loss function is used to measure the discrepancy between the predicted output and the true labels. Common loss functions include mean squared error (MSE) for regression problems and cross-entropy loss for classification problems.

- Backward Pass: The gradients of the loss function with respect to the parameters of the network are computed using the chain rule of derivatives. The gradients indicate how the loss function changes with respect to each parameter, providing information on how to update the parameters to minimize the loss.

- Parameter Update: The gradients computed in the backward pass are used to update the parameters of the network using an optimization algorithm, such as gradient descent or its variants. The optimization algorithm adjusts the weights and biases of the network in the opposite direction of the gradients to minimize the loss function.

- Iteration: Steps 1-4 are repeated for multiple iterations or epochs, where each iteration processes a batch of training data. This allows the network to learn from the entire training dataset and refine its weights and biases.

- Backpropagation is a fundamental component of training neural networks and enables them to learn complex patterns and make accurate predictions. It automates the process of computing gradients, making it efficient to train networks with a large number of parameters.
![Alt text](<project - Copy/Screenshot 2023-05-31 180720.jpg>)

- Below shows the RMSE loss curves on both the training and validation sets with respect to batch index on the x-axis for each optimiser with their optimal hyperparameters respectively.

- The "Train_loss" graph shows the training loss (MSE loss) over the training batches. It provides information about how the loss value changes as the model is trained on different batches of data. This graph helps to monitor the training progress and check if the model is converging or if there are any issues such as overfitting or underfitting.

- The "Val_loss" graph shows the validation loss (MSE loss) over the validation batches. It provides insights into how well the model generalizes to unseen data. By monitoring the validation loss, you can assess the model's performance on data that it has not been directly trained on. A decreasing validation loss indicates that the model is generalizing well, while an increasing validation loss may suggest overfitting or poor generalization.

![Alt text](<project - Copy/Adadelta.jpg>)

![Alt text](<project - Copy/Adagrad.jpg>)

![Alt text](<project - Copy/Adam.jpg>)

![Alt text](<project - Copy/SGD.jpg>)


## Best parameterised network

- Below shows the loss curve for the best parameterised network, we explain the different stages of the loss curve below.

- Initial Sharp Dip (Batch Index 4): The sharp dip in the validation loss indicates that the model initially improved its performance and achieved a lower loss. This can be seen as a positive sign that the model was able to capture some underlying patterns and generalize well to the validation set initially.

- Increase and Fluctuation (Batch Index around 5-56): After the initial dip, the validation loss increases and fluctuates within a certain range. This behavior suggests that the model may have started to overfit to the training data, resulting in higher loss on the validation set. The fluctuations indicate that the model's performance is not consistent across different batches.

- Maximum Loss (Batch Index 56): At a certain point (batch index 56), the validation loss reaches its maximum value of around 1.56e+4. This indicates that the model's performance is at its worst during this phase, likely due to overfitting or an inability to generalize well to the validation data.

- Gradual Decrease (Batch Index 57-149): After reaching the maximum loss, the validation loss gradually decreases. This suggests that the model starts to improve its performance again and generalize better to the validation set. The gradual decrease indicates that the model is adjusting its parameters and learning more meaningful representations that lead to lower loss values.

- Overall, the observed behavior of the validation loss curve indicates some degree of overfitting and fluctuations in the model's performance.
![Alt text](<project - Copy/best_model_loss.jpg>)
- The graph below shows both the loss curves for the training and validation sets of the best parameterised network.

- It is generally expected that the training loss curve would be lower than the validation loss curve during the training process. This is because the model is optimized to minimize the training loss by adjusting its parameters (weights and biases) based on the training data.

- The training loss measures how well the model is fitting the training data. As the model learns from the training examples and iteratively updates its parameters, it tends to improve its performance on the training set. Consequently, the training loss decreases over time, indicating that the model is becoming more accurate in predicting the desired outputs for the training data.

- On the other hand, the validation loss measures how well the model generalizes to unseen data. The validation set is typically used to evaluate the model's performance on data that it has not been directly trained on. If the model becomes too specialized to the training data, it may fail to generalize well to new examples, leading to higher validation loss.
![Alt text](<project - Copy/train_vs_val_best_loss.jpg>)
## Milestone 7- Reusing the NN framework

- In this milestone I reuse my previous neural network framework the only difference is that I Used the load_dataset function to get a new Airbnb dataset where the label is the integer number of bedrooms.

- The previous label (the category) is now included as one of the features.

- I then run the entire pipeline again to train all of the models and find the best one.

- I altered my init method within class AirbnbNightlyPriceImageDataset accordigly while the rest of the previous functions shown in the nueral netwrok stayed the same. 

- To convert the categorical data into a numerical format, the OneHotEncoder class from scikit-learn is used. First, an instance of the OneHotEncoder is created with encoder = OneHotEncoder(). Then, the encoder is fitted to the category_column using encoder.fit(category_column.values.reshape(-1, 1)). This step allows the encoder to learn the unique categories present in the column.

- Transform and concatenate: The category_column is transformed into a one-hot encoded representation using encoder.transform(category_column.values.reshape(-1, 1)). This transformation converts each category into a binary vector. Finally, the one-hot encoded category column is concatenated with the existing features using pd.concat([self.features, pd.DataFrame(category_column.toarray())], axis=1). This ensures that the one-hot encoded category information is included in the dataset features.

- By performing one-hot encoding on the categorical Category column, you convert the categorical variable into a numerical representation that can be used as input for machine learning models. 
```python
""" from sklearn.preprocessing import OneHotEncoder"""
``` 
- The new class is shown below.

```python
""" 
class AirbnbNightlyPriceImageDataset(Dataset):# A class to represent the dataset used in the neural network model

    def __init__(self):
        """_summary_: Selects the necessary features and label from the loaded dataset
        """
        self.data = pd.read_csv('clean_tabular_data.csv')
        self.features, self.label  = load_airbnb(self.data, 'bedrooms')
        category_column = self.data['Category']
        encoder = OneHotEncoder()
        
        # Fit the encoder to your data
        encoder.fit(category_column.values.reshape(-1, 1))
        category_column = encoder.transform(category_column.values.reshape(-1, 1))
        pd.concat([self.features, pd.DataFrame(category_column.toarray())], axis=1)  # Concatenate one-hot encoded category


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
"""
```
## Results
![Alt text](<project - Copy/Adadeltareusecase.jpg>)

![Alt text](<project - Copy/Adagradreusecase.jpg>)

![Alt text](<project - Copy/Adamreusecase.jpg>)

![Alt text](<project - Copy/SGDreusecase.jpg>)
## Best parameterised network
![Alt text](<project - Copy/best_model_loss_reusecase.jpg>)
## Conclusion 
- In conclusion the 
- Continue this process for every milestone, making sure to display clear understanding of each task and the concepts behind them as well as understanding of the technologies used.

- Also don't forget to include code snippets and screenshots of the system you are building, it gives proof as well as it being an easy way to evidence your experience!

## Conclusions

- Maybe write a conclusion to the project, what you understood about it and also how you would improve it or take it further.

- Read through your documentation, do you understand everything you've written? Is everything clear and cohesive?