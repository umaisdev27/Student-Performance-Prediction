# reading the dataset from database
# uploading project to the cloud
# and have all the common things

import os 
import sys
import dill #help to create the pkl file

import numpy as np
import pandas as pd

from src.exception import ExceptionHandling
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise ExceptionHandling(e,sys)
    

def evaluate_models(x_train,y_train,x_test, y_test,models):
    try:
        report = {}

        for model_name, model in models.items(): #items provides both keys and values 
            model.fit(x_train,y_train)

            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)

            r2_score_test = r2_score(y_test,y_pred_test)

            report[model_name] = r2_score_test

            return report

    except Exception as e:
        raise ExceptionHandling(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise ExceptionHandling(e,sys)
