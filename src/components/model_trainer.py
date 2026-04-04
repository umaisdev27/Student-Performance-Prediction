#here we train the different different model
import os
import sys
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.exception import ExceptionHandling
from src.logger import logging
from src.utils import save_object,evaluate_models

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainConfig:
    trained_model_obj_file_path = os.path.join('artifacts','model.pkl')

class ModelTrain:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

    def initiate_model_training(self,train_arr,  test_arr,preprocessor_obj_file):
        try:
            logging.info("Splitting the training and testing Data")

            x_train,y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'random_forest': RandomForestRegressor(),
                'Xg-Boost': XGBRegressor(),
                'Cat-Boost': CatBoostRegressor(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'svm': SVR(),
                'Decision-tree': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'Linear-Regressor': LinearRegression(),
                'AdaBoostRegressor': AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(x_train=x_train,x_test=x_test,
                                              y_train=y_train,y_test=y_test,models=models)

        except Exception as e:
            raise ExceptionHandling(e.sys)
