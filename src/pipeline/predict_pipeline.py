import sys
import pandas as pd

from src.exception import ExceptionHandling
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl',
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path = model_path )

            #the data will go from preprocessor to the model
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise ExceptionHandling(e,sys)

        
#Responsible for mapping all the inputs we are giving to HTML to the BackEnd
#CustomData is called First
class CustomData:
    def __init__(self, gender:str, race_ethnicity:str,lunch:str, parental_level_of_education:str,
                 test_preparation_score:str, reading_score:int, writing_score:int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.lunch = lunch
        self.parental_level_of_education = parental_level_of_education
        self.test_preparation_score = test_preparation_score
        self.reading_score = reading_score
        self.writing_score = writing_score
        # these values are coming from the home.html

    def get_data_as_DataFrame(self):
        try:
            input_data = {
                'gender': [self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'lunch':[self.lunch],
                'parental_level_of_education':[self.parental_level_of_education],
                'test_preparation_score':[self.test_preparation_score],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }

            return pd.DataFrame(input_data)
        
        except Exception as e:
            raise ExceptionHandling(e,sys)

