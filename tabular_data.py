
import pandas as pd
import numpy as np
import os


def remove_rows_with_missing_ratings(data_frame):
    columns_with_na = ['Cleanliness_rating' , 'Accuracy_rating', 'Communication_rating', 'Location_rating',
                                'Check-in_rating', 'Value_rating']
    data_frame.dropna(axis=0, subset = columns_with_na, inplace=True)
    data_frame.drop('Unnamed: 19', axis=1, inplace=True)
    return data_frame

def combine_description_strings(data_frame):
    
    data_frame.dropna(axis=0, subset = 'Description', inplace=True)
    data_frame['Description'] = data_frame['Description'].astype(str)
    data_frame["Description"] = data_frame["Description"].apply(lambda x: x.replace("'About this space', ", '').replace("'', ", '').replace('[', '').replace(']', '').replace('\\n', '. ').replace("''", '').split(" "))
    data_frame["Description"] = data_frame["Description"].apply(lambda x: " ".join(x))
    return data_frame

def set_default_feature_values(data_frame):
    
    list_of_cols=['guests', 'beds', 'bathrooms', 'bedrooms']
    for col in list_of_cols:
        data_frame[col]=data_frame[col].replace(np.nan, 1)

    data_frame=data_frame[data_frame['bedrooms'] != r"https://www.airbnb.co.uk/rooms/49009981?adults=1&category_tag=Tag%3A677&children=0&infants=0&search_mode=flex_destinations_search&check_in=2022-04-18&check_out=2022-04-25&previous_page_section_name=1000&federated_search_id=0b044c1c-8d17-4b03-bffb-5de13ff710bc"]# remove row with url link
    data_frame['bedrooms']=data_frame['bedrooms'].astype(int)
    data_frame['guests']=data_frame['guests'].astype(int)
    return data_frame

def clean_tabular_data(data_frame):
    data= remove_rows_with_missing_ratings(data_frame)
    data2=combine_description_strings(data)
    data3=set_default_feature_values(data2)
    return data3

def load_airbnb(df, label):

        y=df[label]#target/label
        df.drop(label, axis=1, inplace=True)#features
       # text_columns_dataframe=df.select_dtypes('object')
        X= df.select_dtypes(exclude='object')#features
        return (X,y)#tuple of features and target respectively








if __name__ == "__main__" :
    raw_data=pd.read_csv('listing.csv')
    df=clean_tabular_data(raw_data)#processed data
    #processed_data_to_csv=df.to_csv(r"C:\Users\haris\Documents\Aicore\ModellingAirbnbspropertylistingdataset\Modelling-Airbnb-s-property-listing-dataset-\clean_tabular_data.csv")
    #print(df['bathrooms'].dtypes)
    #print(df['Category'])
    #print(clean_tabular_data(df))
  
    # define the file path and name
    # file_path = os.path.join(os.getcwd(), 'clean_tabular_data.csv')

    # # write the DataFrame to a CSV file
    # with open(file_path, 'w',  encoding='utf-8') as f:
    #      df.to_csv(f, index=False)
   # print(load_airbnb(df, label='Price_Night')[1])
    # null_entries = df[df.isin(['NULL'])].any(axis=None)

    # if null_entries:
    #     print('The dataset contains entries with the string "NULL".')
    # else:
    #     print('The dataset does not contain entries with the string "NULL".')

        


    

