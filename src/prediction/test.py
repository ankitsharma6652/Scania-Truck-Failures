import pandas as pd
import numpy as np
import os
import pickle as p
import argparse
import joblib

from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
from src.utils.DbOperations_Logs import DBOperations
from sklearn.preprocessing import StandardScaler


class testPreprocessing:

    def __init__(self, config_path, params_path) -> None:
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.stage_name = os.path.basename(__file__)[:-3]
        self.database_name = self.params['logs_database']['database_name']
        self.prediction_table_name = self.params['logs_database']['prediction_table_name']
        self.user_name = self.config['database']['user_name']
        self.password = self.config['database']['password']
        self.db_logs = DBOperations(self.database_name)
        self.db_logs.establish_connection(self.user_name, self.password)
        self.db_logs.create_table(self.prediction_table_name)
        self.target_columns = self.params['target_columns']['columns']
        # self.scaler_path = self.params['artifacts']['standard_scalar']['standard_scale_file_path']
        
        self.artifacts_dir = self.config["artifacts"]['artifacts_dir']
        self.local_data_dirs = self.config["artifacts"]['local_data_dirs']
        self.local_data_test_file = self.config["artifacts"]['local_data_test_file']
        self.label = self.params["target_columns"]['columns']
        self.scaler_path = self.params['standard_scalar']['standard_scale_file_path']
        self.standard_scale_file_name = self.params['standard_scalar']['standard_scale_file_name']
        

    def get_dataframe(self):
        "This will return dataframe of test dataset."
        try:
            prediction_file_path = os.path.join(self.artifacts_dir, self.local_data_dirs, self.local_data_test_file)
            print(prediction_file_path)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_dataframe",
                                     f"Getting Prediction Dataframe from Dataset")
            return pd.read_csv(prediction_file_path)

        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_dataframe",
                                     f"{e}")
            raise e

    def replaceInvalidValuesWithNull(self,data):

        """ This method replaces invalid values i.e. '?' with null,"""

        # for column in data.columns:
        #     count = data[column][data[column] == '?'].count()
        #     if count != 0:
        #         data[column] = data[column].replace('?', np.nan)

        data.replace('na', np.NaN, inplace=True)

        return data        

    def scale_data(self, data, path, is_dataframe_format_required=False, is_new_scaling=True):
        """
        data: dataframe to perform scaling
        path: path to save scaler object
        get_dataframe_format: default scaled output will be return as ndarray but if is true you will get
        dataframe format
        is_new_scaling: default it will create new scaling object and perform transformation.
        if it is false it will load scaler object from mentioned path paramter
        """
        try:
            path = os.path.join(path)
            if not is_new_scaling:
                if os.path.exists(path):
                    scaler = p.load(os.path.join(path, "standard_scaling.pkl"))
                    output = scaler.transform(data)
                else:
                    raise Exception(f"Scaler object is not found at path: {path}")
            else:
                scaler = StandardScaler()
                output = scaler.fit_transform(data)
                create_directory_path(path)
                joblib.dump(scaler, os.path.join(path, "standard_scaling.pkl"))
            if is_dataframe_format_required:
                output = pd.DataFrame(output, columns=data.columns)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "scale_data",
                                     "Standard Scaling done on Test Dataset")   
            return output
        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "scale_data",
                                     f"{e}")
            raise e        

    def inputFeature_targetFeature(self):
        "This method will seperate input feature dataframe and target feature dataframe."  
        try:
            data_frame = self.get_dataframe()
            
            input_features, target_features = data_frame.drop(self.target_columns, axis=1), data_frame[
                self.target_columns]

            input_features = self.scale_data(data=input_features, path=self.scaler_path,
                                                      is_dataframe_format_required=True,)
            return input_features, target_features

        except Exception as e:
           
            raise e 



if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--params", default="config/params.yaml")

    parsed_args = args.parse_args()

    try:
        preprocessing_object = testPreprocessing(config_path=parsed_args.config, params_path=parsed_args.params)
        preprocessing_object.inputFeature_targetFeature()


    except Exception as e:
        raise e