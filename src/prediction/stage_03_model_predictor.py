import os
import argparse
import joblib
import pandas as pd
import numpy as np
from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
from sklearn.preprocessing import StandardScaler
from src.utils.DbOperations_Logs import DBOperations

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
        self.scaler_path = self.config['artifacts']['training_data']['scaler_path']
        self.model_path = self.config['artifacts']['model']['model_path']
        self.prediction_output_file_path = self.config['artifacts']['prediction_data']['prediction_output_file_path']
        self.prediction_file_name = self.config['artifacts']['prediction_data']['prediction_file_name']

    def get_data(self):
        """This method reads the data from source."""
        self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Entered the get_data method of the predictor class.")
        try:
            prediction_data = pd.read_csv(self.preprocessed_data_path)
            # self.target_column_data = pd.read_csv(self.target_column_data_path).iloc[:,0]
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Data Load Successful.Exited from the get_data method of the predictor class.")
            return prediction_data
        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Exception occured in the predictor class.")
            self-.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Data Load Unsuccessful.Exited from the predictor class.")
            raise e

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
                    scaler = joblib.load(os.path.join(path, "standard_scaling_pred.pkl"))
                    output = scaler.transform(data)
                else:
                    raise Exception(f"Scaler object is not found at path: {path}")
            else:
                scaler = StandardScaler()
                output = scaler.fit_transform(data)
                create_directory_path(path)
                joblib.dump(scaler, os.path.join(path, "standard_scaling_pred.pkl"))
            if is_dataframe_format_required:
                output = pd.DataFrame(output, columns=data.columns)
            return output
        except Exception as e:
            raise e

    def data_preparation(self):
        try:
            input_features = self.get_data()
            # input_features_without_scale = input_features

            input_features = self.scale_data(data=input_features, path=self.scaler_path,
                                                      is_dataframe_format_required=True, is_new_scaling=False)
            return input_features#, input_features_without_scale

        except Exception as e:    
            raise e    

    def get_model_path_list(self):
        try:
            path = os.path.join('config', 'model.yaml')
            config_data = self.read_config(path)
            model_path = []
            for data in config_data['stack']:
                layer = config_data['stack'][data]
                for model in layer:
                    path = f"{data}/{model}"
                    model_path.append(path)
            return model_path
        except Exception as e:
            raise e
    
    def load_model(self):
        try:
            model_path = self.model_path
            model_name = os.listdir(model_path)
            return joblib.load(os.path.join(model_path, model_name))
        except Exception as e:
            raise e

    def predict(self):
        try:
            data = self.data_preparation() 
            model = self.load_model()
            data["Predicted_Feature"] = model.predict(data)      
            prediction_output = data
            create_directory_path(self.prediction_output_file_path)
            output_file_path = os.path.join(self.prediction_output_file_path, self.prediction_file_name)
            if prediction_output is not None:
                prediction_output.to_csv(output_file_path, index=None, header=True)
                self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",f"Prediction file has been generated at {output_file_path}")
        except Exception as e:
            raise e

if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--params", default="config/params.yaml")
    args.add_argument("--model", default="config/model.yaml")

    parsed_args = args.parse_args()

    try:
        predictor = Predictor(config_path=parsed_args.config, params_path=parsed_args.params,model_path=parsed_args.model)
        predictor.predict()
        
    except Exception as e:
        raise e
          

           