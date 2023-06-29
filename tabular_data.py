
import pandas as pd
import numpy as np
import os# import os so I can access me current directory


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

def combine_description_strings(data_frame):
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
    return data_frame

def set_default_feature_values(data_frame):
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
    return data_frame

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
    return data3

def load_airbnb(df, label):
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


def save_clean_csv():
    """_summary_ This function saves the cleaned dataframe as a csv file in the specified file path
    """
    raw_data=pd.read_csv('listing.csv')
    df=clean_tabular_data(raw_data)#processed data
    file_path = os.path.join(os.getcwd(), 'clean_tabular_data.csv')

    # # write the DataFrame to a CSV file
    with open(file_path, 'w',  encoding='utf-8') as f:
         df.to_csv(f, index=False)





if __name__ == "__main__" :

    rdf = pd.read_csv('clean_tabular_data.csv')
    #processed_data_to_csv=df.to_csv(r"C:\Users\haris\Documents\Aicore\ModellingAirbnbspropertylistingdataset\Modelling-Airbnb-s-property-listing-dataset-\clean_tabular_data.csv")
    #print(df['bathrooms'].dtypes)
    #print(df['Category'])
    #print(clean_tabular_data(df))
    print(list(rdf['Category'].unique()))
  
    # define the file path and name
   
   # print(load_airbnb(df, label='Price_Night')[1])
    # null_entries = df[df.isin(['NULL'])].any(axis=None)

    # if null_entries:
    #     print('The dataset contains entries with the string "NULL".')
    # else:
    #     print('The dataset does not contain entries with the string "NULL".')



    

