import os
import argparse
import pandas as pd
import numpy as np
from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
from sklearn.preprocessing import StandardScaler
from src.utils.DbOperations_Logs import DBOperations
import pickle

class Predictor:

    def __init__(self, config_path, params_path, model_path):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.model= read_yaml(model_path)
        self.stage_name = os.path.basename(__file__)[:-3]
        self.database_name = self.params['logs_database']['database_name']
        self.prediction_table_name = self.params['logs_database']['prediction_table_name']
        self.user_name = self.config['database']['user_name']
        self.password = self.config['database']['password']
        self.db_logs = DBOperations(self.database_name)
        self.db_logs.establish_connection(self.user_name, self.password)
        self.db_logs.create_table(self.prediction_table_name)
        self.target_column = self.params["target_columns"]["columns"]
        self.artifacts_dir = self.config["artifacts"]['artifacts_dir']
        self.preprocessed_data_dir = self.config["artifacts"]["preprocessed_data_dir"]
        self.target_column_data_dir = self.config['artifacts']['target_column_data_dir']
        self.preprocessed_test_file = self.config["artifacts"]["preprocessed_test_file"]
        self.target_column_testdata_file = self.config["artifacts"]["target_column_testdata_file"]
        self.model_dir=self.model['model']['model_dir']
        self.preprocessed_data_path = os.path.join(self.artifacts_dir, self.preprocessed_data_dir, self.preprocessed_test_file)
        self.target_column_data_path = os.path.join(self.artifacts_dir, self.target_column_data_dir, self.target_column_testdata_file)
        self.prediction_output_file_path = self.params['artifacts']['prediction_data']['prediction_output_file_path']
        self.prediction_file_name = self.params['artifacts']['prediction_data']['prediction_file_name']
        self.model_path=self.params['artifacts']['model']['model_path']
        self.artifacts_dir = self.config["artifacts"]['artifacts_dir']


    def get_data(self):

        """This method reads the data for prediction from  source."""

        self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Entered the get_data method of the predictor class.")
        try:
            prediction_data = pd.read_csv(self.preprocessed_data_path)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Data Load Successful.Exited from the get_data method of the predictor class.")
            return prediction_data
        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Exception occured in the predictor class.")
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Data Load Unsuccessful.Exited from the predictor class.")
            raise e

    def load_model(self):
        try:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
                                     f"Entered the load_model method of the predictor class")
            model= pickle.load(open(self.model_path,'rb'))
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
                                     f"Model Loaded successfully")
            return model
        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
                                     f"{e}")
            raise e

    def predict(self):
        try:
            data = self.get_data() 
            model = self.load_model()
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",
                                     f"Prediction process started")
            data["Prediction"] = model.predict(data)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",
                                     f"Prediction process completed")
            prediction_output = data
            # create_directory_path(self.prediction_output_file_path)
            create_directory_path([os.path.join(self.artifacts_dir, self.prediction_output_file_path)])

            output_file_path = os.path.join(self.artifacts_dir,self.prediction_output_file_path, self.prediction_file_name)
            # print(output_file_path)
            # print(data)
            if prediction_output is not None:
                prediction_output.to_csv(output_file_path, index=None, header=True)
                self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",f"Prediction file has been generated at {output_file_path}")
                return output_file_path,prediction_output.head(5)
        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",
                                     f"{e}")
            # raise e
            return e


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--params", default="config/params.yaml")
    args.add_argument("--model", default="config/model.yaml")

    parsed_args = args.parse_args()

    try:
        predictor = Predictor(config_path=parsed_args.config, params_path=parsed_args.params,model_path=parsed_args.model)
        predictor.predict()
        predictor.load_model()
    except Exception as e:
        raise e
          

           