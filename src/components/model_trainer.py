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
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainConfig:
    trained_model_obj_file_path = os.path.join('artifacts','model.pkl')

class ModelTrain:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

    def initiate_model_training(self,train_arr,  test_arr):
        try:
            logging.info("Splitting the training and testing Data")

            x_train,y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            #Hyperparameter Tuning Results (ran GridSearchcV -  See hyper_param_tune.ipynb)
            
            models = {
                'random_forest': RandomForestRegressor(),
                'Xg-Boost': XGBRegressor(),
                'Cat-Boost': CatBoostRegressor(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'svm': SVR(),
                'Decision-tree': DecisionTreeRegressor(criterion= 'poisson',
                            max_features= 'log2',
                            min_samples_split = 5,
                            splitter =  'random'),
                            
                'KNeighborsRegressor': KNeighborsRegressor(),
                'Linear-Regressor': LinearRegression(),
                'AdaBoostRegressor': AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(x_train=x_train,x_test=x_test,
                                              y_train=y_train,y_test=y_test,models=models)
            
            #max will sort based on the values
            #this just give the String
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            #it provides the actual model_name
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise ExceptionHandling('No best Model.. Found!!',sys)
            
            logging.info("The best Found model in both training and testing DataSet")

            save_object(
                file_path=self.model_train_config.trained_model_obj_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square
        
        except Exception as e:
            raise ExceptionHandling(e,sys)
