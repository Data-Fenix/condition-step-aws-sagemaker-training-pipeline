"""Evaluation script for measuring model accuracy."""

import json
import logging
import os
import pickle
import tarfile

import pandas as pd
from xgboost import XGBClassifier

##logger = logging.getLogger()
##logger.setLevel(logging.INFO)
##logger.addHandler(logging.StreamHandler())

# May need to import additional metrics depending on what you are measuring.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path="..")

    ##logger.debug("Loading xgboost model.")
    model = pickle.load(open("temp_dict.pkl", "rb"))

    print("Loading validation input data")
    X_val_path = "/opt/ml/processing/X_val/X_val.csv"
    X_val = pd.read_csv(X_val_path, header=None)
    
    X_val.columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                                  'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
                                  'MultipleLines_No phone service', 'MultipleLines_Yes',
                                  'InternetService_Fiber optic', 'InternetService_No',
                                  'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                                  'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                                  'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                                  'TechSupport_No internet service', 'TechSupport_Yes',
                                  'StreamingTV_No internet service', 'StreamingTV_Yes',
                                  'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                                  'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
                                  'PaymentMethod_Credit card (automatic)',
                                  'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
       
    y_val_path = "/opt/ml/processing/y_val/y_val.csv"
    y_val = pd.read_csv(y_val_path, header=None)
    
    y_val.columns = ['Churn']

    #logger.debug("Reading test data.")
    #y_test = df.iloc[:, 0].to_numpy()
    #df.drop(df.columns[0], axis=1, inplace=True)
    #X_test = xgboost.DMatrix(df.values)

    ##logger.info("Performing predictions against test data.")
    predictions = model.predict(X_val)

    print("Creating classification evaluation report")
    acc = accuracy_score(y_val, predictions.round())
    auc = roc_auc_score(y_val, predictions.round())

    # The metrics reported can change based on the model used, but it must be a specific name per (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "auc": {"value": auc, "standard_deviation": "NaN"},
        },
    }

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
