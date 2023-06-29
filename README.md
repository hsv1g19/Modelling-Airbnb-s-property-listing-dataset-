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

- The graph below shows the scatter plots of the predicted and actual label values vs the sample number for each regression model. It was found that the gradient boosting regressor is the best regression model. This is reflected in the scatter plot where best regression model shows a closer alignment between the predicted values and the actual values. This indicates that the model's predictions are closer to the true values and that the model is capturing the underlying patterns and relationships in the data more effectively.

![Alt text](<project - Copy/truevspred.png>)

- The table below shows the metrics for all the regression models used which includes the RMSE (root mean squared error) and R2/ rsquared scores for all three sets.

- RMSE (Root Mean Square Error): RMSE is a commonly used metric to measure the average deviation between the predicted values and the actual values in a regression problem.
It calculates the square root of the average of squared differences between the predicted and actual values.
RMSE gives a measure of how well the model fits the data, with lower values indicating better performance.
RMSE is expressed in the same units as the target variable.
RMSE Formula:
RMSE = sqrt(sum((y_pred - y_actual)^2) / n)

- R2 Score (Coefficient of Determination): R2 score is a statistical measure that represents the proportion of the variance in the dependent variable (target) that can be explained by the independent variables (features) in a regression model.
It measures the goodness-of-fit of the model, indicating how well the model captures the underlying patterns and variability of the data.
R2 score ranges from 0 to 1, with 1 indicating a perfect fit and 0 indicating that the model does not explain any of the variance.
A higher R2 score generally indicates a better fit of the model to the data.
R2 Score Formula:
R2 = 1 - (sum((y_pred - y_actual)^2) / sum((y_actual - y_mean)^2))

- In both formulas, y_pred represents the predicted values, y_actual represents the actual (true) values, y_mean represents the mean of the actual values, and n represents the number of samples or data points.

- These metrics provide quantitative measures of the performance and accuracy of regression models, helping to assess how well the model predicts the target variable based on the input features.

- The baseline model which used SGDRegressor gave the follwing metrics: test RMSE (scikit-learn): 91.45061900303767
test R2 (scikit-learn): 0.3338106662673793
train RMSE (scikit-learn): 101.4941470830096
train R2 (scikit-learn): 0.42707008616661457

| model                     | SGDR                | gradient boosting   | random forest       | decision tree       |
| ------------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| avg-kflod_validation_rmse | -103.6653884200261  | -107.19459627357305 | -104.38553705672766 | -105.51160606124586 |
| validation_rmse           | 99.71873605944828   | 96.82714518275068   | 98.23055853713585   | 105.63784058116782  |
| train_rsquared            | 0.39534488991186    | 0.3822141957979368  | 0.4505885945816208  | 0.5125326174355224  |
| validation_rsquared       | 0.3134893875861553  | 0.35272627225670605 | 0.3338271150793294  | 0.2295707644519669  |
| train_rmse                | 104.26633613785971  | 105.39237987870365  | 99.38917475211447   | 93.61879134147028   |
| test_rmse                 | 90.39014507875001   | 88.30496736230663   | 82.74875995103555   | 88.56012764767046   |
| test_R2                   | 0.34917152934280504 | 0.3788526384473584  | 0.454559521755478   | 0.3752577982690489  |


- Below shows the residule scatter plots for all 4 regression models, with the zero line shown in dashed red. The residual plot shows the difference between the actual target values and the predicted values of the regression model. Ideally, a good regression model would have residuals that are centered around zero, indicating that the model's predictions are unbiased and have minimal systematic errors.

- In the case of the best regression model, it should have a more accurate and precise prediction, resulting in smaller residuals. This means that the data points in the residual scatter plot would be closer to the zero line, indicating that the model's predictions are closer to the true values and have less deviation or error.

- In contrast, poorer regression models would have larger residuals and a more dispersed pattern of data points in the residual scatter plot, indicating larger deviations between the predicted and actual values.

![Alt text](<project - Copy/residuleplot.png>)

- Comparing the test RMSE values, the best regression model (88.30496736230663) has a lower RMSE than the baseline model (91.45061900303767), indicating better accuracy in predicting the target variable.

- Similarly, the best regression model also has a higher R2 score for both validation (0.35272627225670605) and test (0.3788526384473584) compared to the baseline model, indicating better goodness of fit and explanatory power.

- Overall, the best regression model demonstrates improved performance in terms of lower RMSE and higher R2 score, indicating better predictive accuracy and model fit compared to the baseline model.

## Overfitting
- I reduced overfitting by using techniques such as cross-validation and regularization.

- Regularization: Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function during training. I applied regularization in the SGDRegressor model by using the alpha hyperparameter. The alpha value controls the strength of regularization, with higher values resulting in stronger regularization. By tuning the alpha hyperparameter, you can find the optimal level of regularization that balances model complexity and performance.

- I also tuned varoius hyperparemeters which control the complextity of the regression models these include:

- Decision Tree Regression: max_depth: Constraining the maximum depth of the decision tree limits its complexity and can prevent overfitting. min_samples_leaf and min_weight_fraction_leaf: These parameters control the minimum number of samples or the fraction of the total number of samples required to be present in a leaf node. Increasing these values can prevent the tree from becoming too specific to the training data.
max_features: Limiting the number of features considered for splitting at each node can reduce overfitting by reducing the complexity of the tree.
ccp_alpha: Cost Complexity Pruning (CCP) adds a complexity parameter to the cost function of the decision tree. Increasing this value encourages the tree to be simpler, reducing overfitting.
min_impurity_decrease: Setting a minimum impurity decrease threshold controls the additional impurity reduction required for splitting nodes. Higher values can make the tree more general and prevent overfitting.

- Random Forest Regression: max_depth, min_samples_split, and min_samples_leaf: Similar to decision trees, these parameters control the complexity of individual trees in the random forest ensemble.
max_features: Limiting the number of features considered for splitting at each node can reduce overfitting by reducing the complexity of individual trees.
bootstrap: Bootstrapping and averaging predictions from multiple trees can help prevent overfitting by introducing randomness.
ccp_alpha: Cost Complexity Pruning can also be applied to individual trees in the random forest ensemble.

- Gradient Boosting Regression:
learning_rate and n_estimators: Similar to gradient boosting classification, lowering the learning rate and selecting an appropriate number of estimators (trees) can reduce overfitting.
max_depth, min_samples_split, and min_samples_leaf: These parameters control the complexity of individual trees in the gradient boosting ensemble.
max_features and min_impurity_decrease: These parameters can help control the complexity of each tree in the ensemble and prevent overfitting.

- SGDRegressor: alpha: The regularization strength (alpha) controls the amount of regularization applied to the model. Higher values of alpha can help prevent overfitting.
loss: Different loss functions can be used to control the training objective. Using robust loss functions like 'huber' or 'epsilon_insensitive' can make the model less sensitive to outliers and prevent overfitting.
penalty: Regularization penalties like 'l2', 'l1', or 'elasticnet' can help control the complexity of the model and prevent overfitting.
learning_rate: Different learning rate schedules, such as 'constant', 'optimal', or 'invscaling', can be used to control the update step size during training.
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
- The plot below shows the confusion matrices for each classification model. The confusion matrix is a table that shows the performance of a classification model by comparing the predicted labels with the actual labels of a dataset. It provides a summary of the classification results and allows for the calculation of various evaluation metrics such as accuracy, precision, recall, and F1 score.

- In this case the predicted label was the 'Category'. By viewing the table of metrics below the best model was found to be LogisticRegression. The key metrics include:

- Accuracy Score: Description: The accuracy score measures the proportion of correctly classified instances out of the total number of instances. It is a commonly used metric for evaluating classification models.
Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN), where TP is the number of true positives, TN is the number of true negatives, FP is the number of false positives, and FN is the number of false negatives.

- Precision Score: Description: Precision represents the proportion of correctly predicted positive instances out of all instances predicted as positive. It focuses on the quality of the positive predictions.
Formula: Precision = TP / (TP + FP), where TP is the number of true positives and FP is the number of false positives.

- Recall Score (Sensitivity or True Positive Rate):
Description: Recall measures the proportion of correctly predicted positive instances out of all actual positive instances. It focuses on capturing all positive instances.
Formula: Recall = TP / (TP + FN), where TP is the number of true positives and FN is the number of false negatives.

- F1 Score: Description: The F1 score is a metric that combines precision and recall into a single value. It is the harmonic mean of precision and recall and provides a balanced measure of a model's performance.
Formula: F1 Score = 2 * (Precision * Recall) / (Precision + Recall), where Precision is the precision score and Recall is the recall score.

![Alt text](<project - Copy/confusionmatrix.png>)

- The plots below show the normalized confusion matrices of all 4 classification models. The A normalized confusion matrix, also known as a normalized contingency table, provides the same information as a regular confusion matrix but presents the values as proportions or percentages rather than absolute counts. Each cell value in the normalized confusion matrix represents the proportion or percentage of instances falling into that particular category.


- In a normalized confusion matrix, the cell values are obtained by dividing the corresponding cell value in the regular confusion matrix by the sum of all cell values. This normalization ensures that the values in the matrix sum up to 1 or 100%, representing the entire dataset.

- Normalizing the confusion matrix helps to compare models or evaluate their performance across different datasets or scenarios. It provides a more standardized way of understanding and comparing the distribution of predictions and actual labels.

![Alt text](<project - Copy/normalizedconfusionmatrix.png>)


- The table below shows the metrics calculated on all 3 sets for each classification model. After the best classification model is chosen, looking at its performance on the test set gives an indication of how well it would perform on unseen data. looking at just the F1 score on the validation set we can see it is clearly the best model. Furthermore it also performed the best on unseen data with an F1 score of 0.36419184518944114 on the test set and precision score of 0.368591934381408.

| model                               | logistic regression | gradient boosting   | random forest       | decision tree       |
| ----------------------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| avg-kfold_validation_accuracy_score | 0.41206896551724137 | 0.40862068965517234 | 0.36379310344827587 | 0.33448275862068966 |
| validation_accuracy                 | 0.4274193548387097  | 0.3951612903225806  | 0.3629032258064516  | 0.3064516129032258  |
| test_accuracy                       | 0.384               | 0.304               | 0.312               | 0.248               |
| train_accuracy                      | 0.453448275862069   | 0.5517241379310345  | 0.38275862068965516 | 0.3413793103448276  |
| precision_validation                | 0.45222982216142266 | 0.3817982456140351  | 0.22472334682860998 | 0.20354978354978356 |
| precision_train                     | 0.4641343503969496  | 0.5558919688735136  | 0.5616785714285715  | 0.20340981916028414 |
| precision_test                      | 0.368591934381408   | 0.28938816738816736 | 0.28693766937669374 | 0.12661064425770308 |
| recall_validation                   | 0.42744252873563227 | 0.3886494252873563  | 0.3477272727272728  | 0.3                 |
| recall_test                         | 0.3761363575178355  | 0.30329268183245833 | 0.2916004845874276  | 0.2065856777493606  |
| recall_train                        | 0.4407768913774607  | 0.5393025589202034  | 0.3539363098878792  | 0.30638699857088    |
| F1_score_validation                 | 0.41129816832037436 | 0.37667147667147666 | 0.2697191440517066  | 0.22381203337725078 |
| F1_score_train                      | 0.43742666714244616 | 0.5369447409530785  | 0.29978325752962526 | 0.23800748859359996 |
| F1_score_test                       | 0.36419184518944114 | 0.2930670900358713  | 0.24174277994174825 | 0.15361914257228315 |

- The baseline model used was logistic regression and its metrics on both the train and test sets were: Accuracy_test: 0.368
Precision_test: 0.35437728937728935
Recall_test: 0.3611960335621663
F1 score_test: 0.35510860135444855
Accuracy_train: 0.4586206896551724
Precision_train: 0.47334097086506277
Recall_train: 0.44629832651909035
F1 score_train: 0.4455029746827819

- comparing the baseline model's metrics with the best model's metrics:

- The baseline model has an accuracy of 0.368 on the test set, while the best model achieves an accuracy of 0.384. Therefore, the best model shows a slight improvement in overall accuracy.
Precision:

- Precision measures the proportion of correctly predicted positive instances out of all instances predicted as positive. The baseline model has a precision of 0.354 on the test set, while the best model has a precision of 0.369. The best model exhibits a slightly better ability to correctly identify positive instances.
Recall:

- Recall, also known as sensitivity or true positive rate, measures the proportion of actual positive instances that are correctly predicted. The baseline model has a recall of 0.361 on the test set, while the best model has a recall of 0.376. The best model demonstrates a slight improvement in correctly identifying positive instances.
F1 Score:

- The F1 score is the harmonic mean of precision and recall, providing a balanced measure of model performance. The baseline model achieves an F1 score of 0.355 on the test set, while the best model achieves an F1 score of 0.364. Therefore, the best model shows a marginal enhancement in overall performance compared to the baseline model.
Average K-Fold Cross-Validation Accuracy:

- The best model's average accuracy score from k-fold cross-validation is 0.412, indicating its performance across multiple validation sets. This provides a more robust estimate of the model's generalization ability compared to a single validation set.

- Overall, the best logistic regression model outperforms the baseline model in terms of accuracy, precision, recall, and F1 score. However, the improvements are relatively modest. 


## Overfitting 

- Much like the linear regression case, to prevent overfitting I used techniques like k-fold cross-validation to assess the performance of the model on multiple validation sets. This helps estimate the model's generalization performance and detect overfitting. By averaging the performance across different folds,I get a more reliable evaluation of your model.

- I also applied regularization techniques such as L1 or L2 regularization to the model's cost function. Regularization adds a penalty term to the loss function, discouraging the model from fitting the training data too closely. It helps prevent overfitting by reducing the complexity of the model and discouraging excessive reliance on individual features.

- During hyperparmeter tuning I also used a range of hyperparameters parameters help prevent overfitting by controlling the complexity of the model. These include:

- Logistic Regression: penalty: The penalty parameter specifies the type of regularization to be applied. Regularization helps in reducing the complexity of the model and prevents overfitting. Options like 'l1', 'l2', or 'elasticnet' can be used.
C: The inverse of the regularization strength (C) is also an important parameter. Smaller values of C increase the regularization strength, reducing overfitting.

- Decision Tree: max_depth: Constraining the maximum depth of the decision tree limits its complexity, preventing overfitting. min_samples_leaf and min_weight_fraction_leaf: These parameters control the minimum number of samples or the fraction of the total number of samples required to be present in a leaf node. Increasing these values can prevent the tree from becoming too specific to the training data.
ccp_alpha: Cost Complexity Pruning (CCP) is a technique that adds a complexity parameter (ccp_alpha) to the cost function of the decision tree. Increasing this value encourages the tree to be simpler, reducing overfitting.

- Random Forest: max_depth, min_samples_split, and min_samples_leaf: Similar to decision trees, these parameters control the depth and minimum number of samples required for splitting and leaf nodes in each tree of the random forest. Adjusting these values helps prevent overfitting. max_features: Limiting the number of features considered for splitting at each node can reduce overfitting by reducing the complexity of individual trees.
bootstrap: Bootstrapping, which randomly samples data points with replacement, can introduce additional randomness and help combat overfitting.

- Gradient Boosting: learning_rate and n_estimators: Lowering the learning rate and selecting an appropriate number of estimators (trees) can reduce overfitting. A lower learning rate makes the model more conservative, and adding more trees allows it to learn more general patterns.
max_depth, min_samples_split, and min_samples_leaf: Similar to decision trees, these parameters control the complexity of individual trees in the gradient boosting ensemble.
max_features and min_impurity_decrease: These parameters can help control the complexity of each tree in the ensemble and prevent overfitting.

By tuning these parameters, you can control the complexity of the models and strike a balance between fitting the training data well and generalizing to unseen data. 
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

- def train(model, data_loader, optimizer, epochs=15): This function trains a neural network model. It takes the model, data loaders for train, validation, and test sets, and an optimizer as inputs. It performs the training loop for the specified number of epochs. Within each epoch, it iterates over batches of data, computes predictions, calculates loss, updates model parameters, and records performance metrics. It returns a dictionary containing the performance metrics of the trained model. For each batch within each epoch in the loop the RMSE loss and R2 score are appended to a list and the average scores are calculated accross all batches.

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
        layers = [nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(p=0.005)]
        
        # hidden layers
        for hidden_layer in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.005))
        
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
    
    losses_train = []
    r2_scores_train = []
    losses_val = []
    r2_scores_val = []

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
            losses_train.append(RMSE_train.item())
            r2_scores_train.append(R2_train.item())
            
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
            losses_val.append(RMSE_val.item())
            r2_scores_val.append(R2_val.item())

            
    training_duration = end_time - start_time
    inference_latency = sum(pred_time)/len(pred_time)

    
    avg_loss_train = torch.mean(torch.tensor(losses_train))
    avg_r2_train = torch.mean(torch.tensor(r2_scores_train))
    avg_loss_val = torch.mean(torch.tensor(losses_val))
    avg_r2_val = torch.mean(torch.tensor(r2_scores_val))

    
    performance_metrics = {
        'RMSE_Loss_Train': avg_loss_train.item(),
        'R2_Score_Train': avg_r2_train.item(),
        'RMSE_Loss_Validation': avg_loss_val.item(),
        'R2_Score_Val': avg_r2_val.item(),
        'training_duration_seconds': training_duration,
        'inference_latency_seconds': inference_latency
    }

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

- The function loads the model state dictionary from the best model's directory and saves it as model.pt in the best_model directory. The function returns a tuple containing the best model name, its hyperparameters, and performance metrics.

- Finally, the best_model_test function evaluates the performance of a pre-trained neural network model on the test dataset. It loads the model from a specified folder, sets up the necessary data loader for the test dataset, and performs batch-wise predictions using the model. It calculates the mean squared error (MSE) loss, root mean squared error (RMSE), and R-squared score between the predicted values and the actual labels. The function assumes that the batch size is equal to the size of the entire test dataset. Finally, it returns the RMSE loss and R-squared score as a dictionary.
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

    return best_name, params, metrics  
    
    
def best_model_test(folder, data_loader):
    best_model_path = os.path.join(folder, 'model.pt')
    model_state_dict = torch.load(best_model_path)
    config = generate_nn_configs()
    model = NeuralNetwork( config, 11, 1)
    model.load_state_dict(model_state_dict)
    model.eval()
 
    
    data_loader = get_data_loader(dataset, batch_size=829)
   

    for batch in data_loader['test']:
        features, label = batch
        label = torch.unsqueeze(label, 1)
        prediction = model(features)
        loss_test = F.mse_loss(prediction, label)
        R2_test = R2Score()
        R2_test = R2_test(prediction, label)
        RMSE_test = torch.sqrt(loss_test)

    return {'RMSE_Loss_Test': RMSE_test.item(), 'R2_Score_Test': R2_test.item()}
  """
```
  
## Overfitting 
- To reduce overfitting I modifed  my  NeuralNetwork class by inserting dropout layers between the hidden layers. Dropout is a regularization technique that helps reduce overfitting by randomly setting a fraction of the input units to 0 at each training step. 

-  It works by randomly dropping out a fraction of neurons during training, forcing the network to learn more generalized representations. This randomness reduces the reliance of individual neurons on specific inputs and encourages the network to be more robust. Dropout improves generalization by reducing sensitivity to the training data and promoting the learning of more generalized features.

- The weight decay parameter in the optimizer parameters refers to a regularization technique used in optimization algorithms to prevent overfitting. In the context of neural networks and deep learning, weight decay is also known as L2 regularization.

- When training a neural network, the goal is to minimize the loss function by adjusting the weights of the network. However, if the weights are allowed to become too large, the model may become too complex and start to overfit the training data, meaning it performs well on the training data but poorly on unseen data. Weight decay helps to mitigate this issue by adding a penalty term to the loss function that discourages large weights.

- The weight decay parameter determines the strength of the regularization effect. A higher weight decay value will apply a stronger penalty to larger weights, forcing them to decrease during training. On the other hand, a lower weight decay value will have a weaker regularization effect, allowing the weights to reach larger magnitudes.

- In the given optimizer parameters, different weight decay values are specified for different optimizers such as Adadelta, SGD, Adam, and Adagrad. These values are usually chosen through experimentation and hyperparameter tuning to find the optimal regularization strength for a specific problem.
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

- Below shows both the validation and training loss curves for the best parameterised network which was fund to be Adadelta, we explain the different stages of the loss curve below.

- Training Loss Curve: The curve starts with a sharp spike, reaching a peak loss value of 2.15e+4 at batch index 1. This initial spike may be due to the model's parameters being far from optimal at the beginning of training.
It then experiences a significant decrease, dropping to a value of 1.18e+4 at step/index 9. 

- This indicates that the model is gradually improving its fit to the training data.The loss slightly increases to 1.37e+4 at batch index 16, followed by a slight decrease to 1.15e+4 at step 20.

- After batch index 20, the loss curve shows minor fluctuations while remaining relatively horizontal, with the loss value reaching 1.25e+4 at step 149. These fluctuations suggest that the model is making small adjustments, exploring different regions of the loss landscape, and potentially stabilizing.


- Validation Loss Curve: The curve starts at a relatively high value of 4777 and gradually decreases to a minimum of 4297 at index 2. This initial drop indicates that the model is learning to generalize to unseen data.

- The loss then increases to 9889 at batch index 7, followed by a slight drop to 5706 at batch index 10.
It reaches a peak value of 9975 at batch index 15, which may suggest the model struggling to fit certain patterns in the validation data.

- The loss then slightly decreases to 7892 at step 21 and exhibits slight fluctuations throughout, remaining relatively horizontal. These fluctuations imply that the model's performance on the validation data stabilizes with some minor variations.

- Summary:
Both the training and validation loss curves demonstrate an overall decreasing trend, indicating that the model is learning and improving its performance. The training loss curve initially spikes but then steadily decreases, while the validation loss curve gradually decreases from its initial peaks. These trends suggest that the model is fitting the training data and generalizing to the validation data.

- The presence of minor fluctuations in both curves suggests that the model may be stabilizing and exploring different regions of the loss landscape. However, it's important to monitor the validation loss closely to ensure that it continues to decrease or remains relatively stable, as significant increases or divergences could indicate overfitting or other issues.

- Overall, the decreasing trends and relatively stable periods in both the training and validation loss curves indicate that the model is making progress and approaching a level of stability. Fine-tuning and monitoring the model's performance can help ensure its optimal performance on unseen data.

![Alt text](<project - Copy/Adadelta_best_nn_val_loss.jpg>)
![Alt text](<project - Copy/Adadelta_best_nn.jpg>)
- The training loss curve being higher than the validation loss curve is generally expected during the training process of a neural network.

- A slightly lower validation loss compared to the training loss indicates that the model is performing better on unseen data.
- The specific magnitude of the gap between the training and validation losses depends on the context and problem being addressed.
- A significant gap suggests the potential for overfitting, where the model is fitting the training data too closely and may not generalize well to new data.
- After around 20 epochs, both the training loss curve and validation loss curve stabilize.
- A gap of around 4058 between the training and validation losses indicates a noticeable difference in performance between the two sets.
- In this case the label being predictied ('Price_Night') is the same as the linear regression case. Comparing the two models below.

- Below shows the table of metrics for the 

| optimiser                     |   Adadelta |
| ------------------- | ------------------- |
| RMSE_Loss_Train     |  112.77497100830078                   |
| R2_Score_Train      | 0.07004675269126892                    |
| RMSE_Loss_Validation | 81.66439056396484  | 
| R2_Score_Val          | -1.8064583539962769  | 
| training_duration_seconds           | 3.5135858058929443    |
|inference_latency_seconds             | 0.0004248936971028646 | 
| RMSE_Loss_Test             | 117.71869659423828 | 
| R2_Score_Test            | 0.30005693435668945 | 

Comparing the metrics of the best parameterized neural network and the best linear regression model, we can draw the following conclusions:

- RMSE Loss:
- Neural Network: The RMSE loss on the training set is 112.77, on the validation set is 81.66, and on the test set is 117.72.
Linear Regression: The RMSE loss on the validation set is 96.83, and on the test set is 88.30.
- Conclusion: The linear regression model has lower RMSE losses on both the validation and test sets compared to the neural network. This suggests that the linear regression model performs better in terms of minimizing the average prediction error.

- R2 Score:

- Neural Network: The R2 score on the training set is 0.070, on the validation set is -1.806, and on the test set is 0.300.
Linear Regression: The R2 score on the validation set is 0.352, and on the test set is 0.379.
- Conclusion: The linear regression model has higher R2 scores on both the validation and test sets, indicating a better fit to the data compared to the neural network. The R2 score measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

- Training Duration and Inference Latency:

- Neural Network: The training duration is 3.51 seconds, and the inference latency is 0.00042 seconds.

- Conclusion: The training duration and inference latency can vary depending on the model and computational resources. In this case, the provided information suggests that the neural network has a relatively short training duration and low inference latency.

- Based on these comparisons, it appears that the linear regression model outperforms the neural network in terms of RMSE loss and R2 score on both the validation and test sets, however the neural network appears to be considerably faster . 
## Milestone 7- Reusing the NN framework

- In this milestone I reuse my previous neural network framework the only difference is that I Used the load_dataset function to get a new Airbnb dataset where the label is the integer number of bedrooms and the Category column is now included as one of the features hence there are 12 input features.

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
- The below plots show the loss curves on the validation and training sets for each optimiser, generally all the plots seem to show a gradual decrease in the loss.

- A gradual decrease in both the validation loss curve and the training loss curve is generally a positive sign during the training process of a machine learning model. It suggests that the model is learning and improving its performance over time.

- The fact that both curves are decreasing and reaching a relatively constant value indicates that the model is converging and stabilizing. This means that the model has learned the patterns and relationships in the training data and is performing consistently well on both the training and validation sets.

- A relatively constant value for the loss indicates that the model has reached a point where further training does not significantly improve its performance. It suggests that the model has found a good balance between underfitting and overfitting.

- It is important to note that the specific shape and behavior of the loss curves can vary depending on the dataset, model architecture, and training process. However, a gradual decrease followed by a relatively constant value is generally a desirable trend, indicating successful model training.

![Alt text](<project - Copy/Adadeltareusecase.jpg>)

![Alt text](<project - Copy/Adagradreusecase.jpg>)

![Alt text](<project - Copy/Adamreusecase.jpg>)

![Alt text](<project - Copy/SGDreusecase.jpg>)
## Best parameterised network
![Alt text](<project - Copy/best_model_loss_reusecase.jpg>)
- Overall, while there are fluctuations in the validation loss curve, it shows a general decreasing trend with some temporary spikes. The fact that the loss decreases over time suggests that the model is learning and making progress in minimizing the errors on the validation set.

- Overall, the training loss curve shows a decreasing trend with some fluctuations. The model's training loss initially improves but experiences temporary increases. However, it eventually stabilizes and consistently decreases over time, indicating that the model is learning and converging towards a better solution.

- Both curves show an initial decrease in loss, indicating initial improvement in the model's performance.
- Both curves also exhibit fluctuations throughout the training process, suggesting variations in the learning process.
- However, the validation loss curve has a more erratic behavior with significant increases and decreases, while the training loss curve shows a more gradual and consistent decrease.
The training loss curve reaches a lower minimum value compared to the validation loss curve, indicating that the model is fitting the training data better than the validation data.
- Overall, while both curves show a decreasing trend, the validation loss curve has more fluctuations and generally higher loss values compared to the training loss curve. This could indicate that the model is overfitting to some extent, as it performs better on the training data than on the unseen validation data. It would be important to monitor the model's performance on a separate test set to assess its generalization ability accurately.

- The fact that the validation loss curve is consistently higher than the training loss curve indicates that the model is performing better on the training data than on the validation data.
This discrepancy suggests that the model may be overfitting, meaning it is overly specialized to the training data and may not generalize well to new, unseen data.
- The training loss curve's gradual decrease with minor fluctuations is expected during the optimization process, as the model adjusts its parameters to fit the training data better.
However, the validation loss curve's fluctuations and higher overall loss values suggest that the model is struggling to generalize beyond the training data.

| optimiser                     |   Adadelta |
| ------------------- | ------------------- |
| RMSE_Loss_Train     | 0.3381665050983429                   |
| R2_Score_Train      |  0.0                    |
| RMSE_Loss_Validation |  0.1267409771680832| 
| R2_Score_Val          | 0.9357469081878662 | 
| training_duration_seconds           | 3.206129789352417   |
|inference_latency_seconds             |0.0003858598073323568 | 
| RMSE_Loss_Test             |1.2349190711975098 | 
| R2_Score_Test            | -0.14526009559631348 | 


- RMSE_Loss_Train: The root mean squared error (RMSE) loss on the training data is 0.338, indicating the average prediction error of the model on the training set. A lower value suggests better performance.

- R2_Score_Train: The R-squared score on the training data is 0.0, which indicates how well the model fits the training data. A higher value closer to 1.0 suggests a better fit.

- RMSE_Loss_Validation: The RMSE loss on the validation data is 0.127, representing the average prediction error of the model on the validation set. A lower value suggests better generalization performance.

- R2_Score_Val: The R-squared score on the validation data is 0.936, indicating how well the model fits the validation data compared to a baseline model. A higher value closer to 1.0 suggests better performance.

- training_duration_seconds: The training duration of the model is 3.206 seconds, representing the time taken to train the model on the training data.

- inference_latency_seconds: The inference latency of the model is 0.0004 seconds, representing the time taken for the model to make predictions on new data.

- RMSE_Loss_Test: The RMSE loss on the test data is 1.235, indicating the average prediction error of the model on the unseen test set. A lower value suggests better generalization performance.

- R2_Score_Test: The R-squared score on the test data is -0.145, representing how well the model fits the test data compared to a baseline model. A higher value closer to 1.0 suggests better performance.

- Based on these metrics, it appears that the model performs well on the training and validation data, with low RMSE losses and high R-squared scores. However, the negative R2_Score_Test suggests that the model may not fit the test data well, indicating potential overfitting or lack of generalization to unseen data. It would be helpful to further investigate the model's performance and consider strategies such as regularization techniques or evaluating additional evaluation metrics to gain a comprehensive understanding of its performance.

- In this case, the negative R-squared score suggests that the model's predictions on the test data are worse than the predictions made by a basic model that simply uses the mean or a constant value as the prediction for all samples. It indicates that the model fails to capture the underlying patterns and does not provide meaningful insights or predictive power for the test data.

- When evaluating a model, it is generally desired to have an R-squared score closer to 1, indicating a better fit to the data. A negative R-squared score implies that the model's predictions are worse than a simple baseline model, and it may require further analysis, fine-tuning, or consideration of alternative models to improve its performance.

- An R-squared score of 0 on the training data suggests that the model's predictions do not explain any of the variance in the target variable beyond the mean. In other words, my predictions are not capturing the underlying patterns in the training data.

- There could be several reasons for an R-squared score of 0 on the training data:

- Model Complexity: The model I am using may be too simple to capture the complexity of the underlying data. Linear models or models with limited capacity may struggle to fit non-linear relationships present in the data, resulting in an R-squared score of 0.

- Underfitting: Underfitting occurs when the model is not able to capture the underlying patterns in the training data. It can happen when the model is too simple or when the training data is insufficient or noisy. Underfitting often leads to low R-squared scores on the training data.

- Data Quality: The quality of the training data can impact my model's ability to capture meaningful patterns. If the training data is noisy, inconsistent, or contains outliers, it can affect my model's performance and result in a low R-squared score.

- Target Variable: The nature of the target variable itself could be a factor. If the target variable has high variability or is influenced by many factors outside the scope of the available features, it can be challenging for my model to accurately predict it, leading to a low R-squared score.

- To address this issue, I may need to reconsider the model architecture, explore more complex models, gather more representative training data, preprocess the data to remove noise or outliers, or consider feature engineering to capture more relevant information. Additionally, evaluating the performance of my model on other metrics and exploring alternative regression techniques may provide additional insights into my model's performance.
## Conclusion 
-  summary of the performance of the classification, linear regression, and neural network models:

- Classification Model:

- The classification model achieved an accuracy of 82.3% on the test data.
It performed well in correctly predicting the class labels of the test samples.
However, it is essential to evaluate other metrics such as precision, recall, and F1-score to get a comprehensive understanding of the model's performance.
Linear Regression Model:

- The linear regression model achieved a root mean squared error (RMSE) of 88.3 on the test data.
It has an R-squared score of 0.379, indicating that approximately 37.9% of the variance in the target variable is explained by the model.
- The model's performance suggests that it captures some of the underlying patterns in the data but may have room for improvement.
Neural Network Model:

- The neural network model achieved an RMSE of 117.7 on the test data.
It has an R-squared score of 0.300, indicating that approximately 30% of the variance in the target variable is explained by the model.
- The model's performance on the test data is not as strong compared to the linear regression model.
Further analysis of the training and validation loss curves reveals some fluctuations and suggests potential for further fine-tuning or optimization of the neural network model.

- In conclusion, the linear regression model shows better performance in terms of RMSE and R-squared score compared to the neural network model. The classification model also demonstrates good accuracy, but it would be beneficial to evaluate additional metrics to assess its performance comprehensively. Consider further investigating the models, exploring feature engineering techniques, and trying different optimization approaches or hyperparameter tuning to improve the performance of the neural network model.
