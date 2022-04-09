import boto3
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import argparse
import os
import warnings
warnings.simplefilter(action='ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def change_format(df):
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    
    return df

def missing_value(df):
    print("count of missing values: (before treatment)", df.isnull().sum())
    
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    print("count of missing values: (before treatment)", df.isnull().sum())
    print("missing values successfully replaced")
    return df

def data_manipulation(df):
    df = df.drop(['customerID'], axis = 1)
    
    return df

def cat_encoder(df, variable_list):
    dummy = pd.get_dummies(df[variable_list], drop_first = True)
    df = pd.concat([df, dummy], axis=1)
    df.drop(df[cat_var], axis = 1, inplace = True)
    
    print("Encoded successfully")
    return df

def scaling(X):  
    min_max=MinMaxScaler()
    X=pd.DataFrame(min_max.fit_transform(X),columns=X.columns)
    
    return X

if __name__ == "__main__":


    input_data_path = os.path.join("/opt/ml/processing/input", 'telco_cutomer_churn.csv')


    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    
    columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
           'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

    df.columns = columns

    cat_var = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'Churn']
    
    df = data_manipulation(missing_value(change_format(df)))
    df = cat_encoder(df, cat_var)

    X = df.iloc[:, 0:30]
    y = df.iloc[:, -1]
    X = scaling(X)
    
    print("split the dataset into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
        
    print("split the dataset into test and validation sets")
    # Use the same function above for the validation set
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, 
            test_size=0.4, random_state= 5)

    print("X_train shape: {}".format(X_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("y_test shape: {}".format(y_test.shape))
    print("X_val shape: {}".format(X_val.shape))
    print("y_val shape: {}".format(y_val.shape))
    
    print("Saving the outputs")
    X_output_path = os.path.join("/opt/ml/processing/output1", "X.csv")   
        
    print("Saving output to {}".format(X_output_path))
    pd.DataFrame(X).to_csv(X_output_path, header=False, index=False)
    
    y_output_path = os.path.join("/opt/ml/processing/output2", "y.csv")   
        
    print("Saving output to {}".format(y_output_path))
    pd.DataFrame(y).to_csv(y_output_path, header=False, index=False)
    
    print("Saving the outputs for evaluation script")
    X_val_path = os.path.join("/opt/ml/processing/X_val", "X_val.csv")
    print("Saving output to {}".format(X_val_path))
    pd.DataFrame(X_val).to_csv(X_val_path, header=False, index=False)
    
    y_val_path = os.path.join("/opt/ml/processing/y_val", "y_val.csv")
    print("Saving output to {}".format(y_val_path))
    pd.DataFrame(y_val).to_csv(y_val_path, header=False, index=False)
