# reading the dataset from database
# uploading project to the cloud
# and have all the common things

import os 
import sys
import dill #help to create the pkl file

import numpy as np
import pandas as pd

from src.exception import ExceptionHandling

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise ExceptionHandling(e,sys)
