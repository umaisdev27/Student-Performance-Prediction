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
from src.utils import save_object

# Provide any path or input for Data Transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """This Function is responsible for the Data_Transformation"""
        try:
            #first Create a numerical and categorical features
            numerical_cols = ['writing_score','reading_score']
            categorical_features = ['gender','parental_level_of_education',
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
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise ExceptionHandling(e,sys)  
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Complete Reading the Train and Test Data")

            logging.info("Obtaining Preprocessor Object")

            preprocessing_obj = self.get_data_transformer_obj()
            
            target_column_names = 'math_score'
            numerical_cols = ['writing_score','reading_score']

            #Train Features
            input_feature_train_df = train_df.drop(columns=target_column_names)
            target_feature_train_df = train_df[target_column_names]

            #Test Features
            input_feature_test_df = test_df.drop(columns=target_column_names)
            target_feature_test_df = test_df[target_column_names]

            logging.info('Apllying the Preprocessing object in the ' \
                    'Train DataFrame and Test DataFrame')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.fit(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(input_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(input_feature_test_df)
            ]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr,  test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            raise ExceptionHandling(e,sys)