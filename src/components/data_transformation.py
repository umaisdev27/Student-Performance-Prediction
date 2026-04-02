import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import ExceptionHandling
from src.logger import logging

# Provide any path or input for Data Transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            #first Create a numerical and categorical features
            numerical_cols = ['writing_score','reading_score','math_score']
            categorica_features = ['gender','parental_level_of_education',
                                   'lunch','test_preparation_course',
                                   'race_ethnicity']
            
            #2nd: Create the Pipeline for num and cat
            num_pipeline = Pipeline(
            steps = [
                ("Imputer",SimpleImputer(strategy='median')),
                ("StandardScaler",StandardScaler())
                ]
            )

            logging.info('Numerical Features Scaling Completed')

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder',OneHotEncoder()),
                    ("StandardScaler",StandardScaler())
                ]
            )

            logging.info('Categorical Columns Encoding Completed')

            #Combine these Pipelines together through the ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,cat_pipeline)
                ]
            )


        except Exception as e:
            raise ExceptionHandling(e,sys)  