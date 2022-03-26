# import os
# import argparse
# import joblib
# import pandas as pd
# import numpy as np
# from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
# from sklearn.preprocessing import StandardScaler
# from src.utils.DbOperations_Logs import DBOperations
# import pickle
#
#
# class Predictor:
#
#     def __init__(self, config_path, params_path, model_path):
#         self.config = read_yaml(config_path)
#         self.params = read_yaml(params_path)
#         self.model = read_yaml(model_path)
#         self.stage_name = os.path.basename(__file__)[:-3]
#         self.database_name = self.params['logs_database']['database_name']
#         self.prediction_table_name = self.params['logs_database']['prediction_table_name']
#         self.user_name = self.config['database']['user_name']
#         self.password = self.config['database']['password']
#         self.db_logs = DBOperations(self.database_name)
#         self.db_logs.establish_connection(self.user_name, self.password)
#         self.db_logs.create_table(self.prediction_table_name)
#         self.target_column = self.params["target_columns"]["columns"]
#         self.artifacts_dir = self.config["artifacts"]['artifacts_dir']
#         self.preprocessed_data_dir = self.config["artifacts"]["preprocessed_data_dir"]
#         self.target_column_data_dir = self.config['artifacts']['target_column_data_dir']
#         self.preprocessed_test_file = self.config["artifacts"]["preprocessed_test_file"]
#         self.target_column_testdata_file = self.config["artifacts"]["target_column_testdata_file"]
#         self.model_dir = self.model['model']['model_dir']
#         self.preprocessed_data_path = os.path.join(self.artifacts_dir, self.preprocessed_data_dir,
#                                                    self.preprocessed_test_file)
#         self.target_column_data_path = os.path.join(self.artifacts_dir, self.target_column_data_dir,
#                                                     self.target_column_testdata_file)
#         # self.scaler_path = self.config['artifacts']['training_data']['scaler_path']
#         # self.model_path = self.config['artifacts']['model']['model_path']
#         self.prediction_output_file_path = self.params['artifacts']['prediction_data']['prediction_output_file_path']
#         self.prediction_file_name = self.params['artifacts']['prediction_data']['prediction_file_name']
#         self.model_path = self.params['artifacts']['model']['model_path']
#         self.artifacts_dir = self.config["artifacts"]['artifacts_dir']
#
#     def get_data(self):
#         """This method reads the data for prediction from  source."""
#         self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data",
#                                  f"Entered the get_data method of the predictor class.")
#         try:
#             prediction_data = pd.read_csv(self.preprocessed_data_path)
#             # self.target_column_data = pd.read_csv(self.target_column_data_path).iloc[:,0]
#             self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data",
#                                      f"Data Load Successful.Exited from the get_data method of the predictor class.")
#             return prediction_data
#         except Exception as e:
#             self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data",
#                                      f"Exception occured in the predictor class.")
#             self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data",
#                                      f"Data Load Unsuccessful.Exited from the predictor class.")
#             raise e
#
#     # def scale_data(self, data, path, is_dataframe_format_required=False, is_new_scaling=True):
#     #     """
#     #     data: dataframe to perform scaling
#     #     path: path to save scaler object
#     #     get_dataframe_format: default scaled output will be return as ndarray but if is true you will get
#     #     dataframe format
#     #     is_new_scaling: default it will create new scaling object and perform transformation.
#     #     if it is false it will load scaler object from mentioned path paramter
#     #     """
#     #     try:
#     #         path = os.path.join(path)
#     #         if not is_new_scaling:
#     #             if os.path.exists(path):
#     #                 scaler = joblib.load(os.path.join(path, "standard_scaling_pred.pkl"))
#     #                 output = scaler.transform(data)
#     #             else:
#     #                 raise Exception(f"Scaler object is not found at path: {path}")
#     #         else:
#     #             scaler = StandardScaler()
#     #             output = scaler.fit_transform(data)
#     #             create_directory_path(path)
#     #             joblib.dump(scaler, os.path.join(path, "standard_scaling_pred.pkl"))
#     #         if is_dataframe_format_required:
#     #             output = pd.DataFrame(output, columns=data.columns)
#     #         return output
#     #     except Exception as e:
#     #         raise e
#
#     # def data_preparation(self):
#     #     try:
#     #         input_features = self.get_data()
#     #         # input_features_without_scale = input_features
#
#     #         input_features = self.scale_data(data=input_features, path=self.scaler_path,
#     #                                                   is_dataframe_format_required=True, is_new_scaling=False)
#     #         return input_features#, input_features_without_scale
#
#     #     except Exception as e:
#     #         raise e
#
#     # def get_model_path_list(self):
#     #     try:
#     #         path = os.path.join('config', 'model.yaml')
#     #         config_data = self.read_config(path)
#     #         model_path = []
#     #         for data in config_data['stack']:
#     #             layer = config_data['stack'][data]
#     #             for model in layer:
#     #                 path = f"{data}/{model}"
#     #                 model_path.append(path)
#     #         return model_path
#     #     except Exception as e:
#     #         raise e
#
#     def load_model(self):
#         try:
#             # model_path = r'artifacts/model_dir'
#             model_name = os.listdir(self.model_path)
#             # print(*model_name)
#             model_path = (os.path.join(self.model_path, *model_name))
#             self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
#                                      f"Entered the load_model method of the predictor clas")
#             model = pickle.load(open(model_path, 'rb'))
#             self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
#                                      f"Model Loaded successfully")
#             return model
#         except Exception as e:
#             self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
#                                      f"{e}")
#             print(e)
#             raise e
#
#     def predict(self):
#         try:
#             data = self.get_data()
#             model = self.load_model()
#             self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",
#                                      f"Prediction process started")
#             data["Prediction"] = model.predict(data)
#             self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",
#                                      f"Prediction process completed")
#             prediction_output = data
#             # create_directory_path(self.prediction_output_file_path)
#             create_directory_path([os.path.join(self.artifacts_dir, self.prediction_output_file_path)])
#
#             output_file_path = os.path.join(self.artifacts_dir, self.prediction_output_file_path,
#                                             self.prediction_file_name)
#             # print(output_file_path)
#             # print(data)
#             if prediction_output is not None:
#                 prediction_output.to_csv(output_file_path, index=None, header=True)
#                 self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",
#                                          f"Prediction file has been generated at {output_file_path}")
#                 return output_file_path, prediction_output.head(5)
#         except Exception as e:
#             self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",
#                                      f"{e}")
#             # raise e
#             return e
#
#
# if __name__ == '__main__':
#
#     args = argparse.ArgumentParser()
#
#     args.add_argument("--config", default="config/config.yaml")
#     args.add_argument("--params", default="config/params.yaml")
#     args.add_argument("--model", default="config/model.yaml")
#
#     parsed_args = args.parse_args()
#
#     try:
#         predictor = Predictor(config_path=parsed_args.config, params_path=parsed_args.params,
#                               model_path=parsed_args.model)
#         predictor.predict()
#         # predictor.load_model()
#     except Exception as e:
#         raise e

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
from sklearn.preprocessing import StandardScaler
from src.utils.DbOperations_Logs import DBOperations
import pickle
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
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
        # self.scaler_path = self.config['artifacts']['training_data']['scaler_path']
        # self.model_path = self.config['artifacts']['model']['model_path']
        self.prediction_output_file_path = self.params['artifacts']['prediction_data']['prediction_output_file_path']
        self.prediction_file_name = self.params['artifacts']['prediction_data']['prediction_file_name']
        self.model_path=self.params['artifacts']['model']['model_path']
        self.random_forest=self.model['model']['random_forest']
        self.xgboost=self.model['model']['xgboost']
        self.artifacts_dir = self.config["artifacts"]['artifacts_dir']
        self.db_logs.best_model_table()
        access_key, secret_access_key = self.db_logs.get_aws_s3_keys()
        self.aws = AmazonSimpleStorageService(access_key, secret_access_key, self.config['storage']['bucket_name'])

    def get_data(self):
        """This method reads the data for prediction from  source."""
        self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Entered the get_data method of the predictor class.")
        try:
            # prediction_data = pd.read_csv(self.preprocessed_data_path)
            prediction_data=self.aws.read_csv_file(os.path.join(self.artifacts_dir, self.preprocessed_data_dir).replace("\\","/"),self.preprocessed_test_file)['data_frame']
            print("File loaded successfully")
            # self.target_column_data = pd.read_csv(self.target_column_data_path).iloc[:,0]
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Data Load Successful from S3 STorage.Exited from the get_data method of the predictor class.")
            return prediction_data
        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Exception occured in the predictor class.")
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_data", f"Data Load Unsuccessful.Exited from the predictor class.")
            raise e
            return e

    # def scale_data(self, data, path, is_dataframe_format_required=False, is_new_scaling=True):
    #     """
    #     data: dataframe to perform scaling
    #     path: path to save scaler object
    #     get_dataframe_format: default scaled output will be return as ndarray but if is true you will get
    #     dataframe format
    #     is_new_scaling: default it will create new scaling object and perform transformation.
    #     if it is false it will load scaler object from mentioned path paramter
    #     """
    #     try:
    #         path = os.path.join(path)
    #         if not is_new_scaling:
    #             if os.path.exists(path):
    #                 scaler = joblib.load(os.path.join(path, "standard_scaling_pred.pkl"))
    #                 output = scaler.transform(data)
    #             else:
    #                 raise Exception(f"Scaler object is not found at path: {path}")
    #         else:
    #             scaler = StandardScaler()
    #             output = scaler.fit_transform(data)
    #             create_directory_path(path)
    #             joblib.dump(scaler, os.path.join(path, "standard_scaling_pred.pkl"))
    #         if is_dataframe_format_required:
    #             output = pd.DataFrame(output, columns=data.columns)
    #         return output
    #     except Exception as e:
    #         raise e

    # def data_preparation(self):
    #     try:
    #         input_features = self.get_data()
    #         # input_features_without_scale = input_features

    #         input_features = self.scale_data(data=input_features, path=self.scaler_path,
    #                                                   is_dataframe_format_required=True, is_new_scaling=False)
    #         return input_features#, input_features_without_scale

    #     except Exception as e:
    #         raise e

    # def get_model_path_list(self):
    #     try:
    #         path = os.path.join('config', 'model.yaml')
    #         config_data = self.read_config(path)
    #         model_path = []
    #         for data in config_data['stack']:
    #             layer = config_data['stack'][data]
    #             for model in layer:
    #                 path = f"{data}/{model}"
    #                 model_path.append(path)
    #         return model_path
    #     except Exception as e:
    #         raise e

    def load_model(self):
        try:
            # model_path = r'artifacts/model_dir'
            # print(self.model_path)
            # model_name = os.listdir(self.model_path)
            # print(model_name)
            # model_path=(os.path.join(self.model_path, model_name))
            # print(model_path)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
                                     f"Entered the load_model method of the predictor class")
            get_model_name_from_db=self.db_logs.get_best_model_name()
            # print(get_model_name_from_db)
            # print(self.model_path)
            # if get_model_name_from_db+".pkl" in os.listdir(self.model_path):
            #
            #     model= pickle.load(open(os.path.join(self.model_path,get_model_name_from_db+".pkl"),'rb'))
            #     self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
            #                          f"Model Loaded successfully")
            #     return model
            if self.aws.is_file_present(self.model_path,get_model_name_from_db+".pkl")['status']:
                print("File present in AWS S3 Storage-Load_model()")

                model= pickle.load(open(os.path.join(self.model_path,get_model_name_from_db+".pkl"),'rb'))
                model=self.aws.get_pickle_file(self.model_path,get_model_name_from_db+".pkl")['file_content']
                self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
                                     f"Model Loaded successfully")
                return model
            else:
                self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
                                         f"Model not Loaded successfully")
        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "load_model",
                                     f"{e}")
            raise e
            return e

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
            # if prediction_output is not None:
            self.aws.write_file_content(os.path.join(self.artifacts_dir,self.prediction_output_file_path).replace("\\","/"),self.prediction_file_name,prediction_output)
            prediction_output.to_csv(output_file_path, index=None, header=True)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",f"Prediction file has been generated at {output_file_path}")
            print("Prediction completed")
            self.aws.upload_file(os.path.join(self.artifacts_dir,self.prediction_output_file_path).replace("\\","/"),self.prediction_file_name,self.prediction_file_name,local_file_path=output_file_path,over_write=False)
            print("Prediction File uploaded to AWS S3 storage")
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",f"Prediction file has been generated at S3 Storage Location {output_file_path}")

            return output_file_path,prediction_output
        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "predict",
                                     f"{e}")
            # raise e
            return e,e
    def download_prediction_file(self):
        return self.aws.download_file(os.path.join(self.artifacts_dir,self.prediction_output_file_path).replace("\\","/"),self.prediction_file_name,local_system_directory=r"D:\CloudStorageAutomation\cloud_storage_layer")

if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--params", default="config/params.yaml")
    args.add_argument("--model", default="config/model.yaml")

    parsed_args = args.parse_args()

    try:
        predictor = Predictor(config_path=parsed_args.config, params_path=parsed_args.params,model_path=parsed_args.model)
        predictor.predict()
        # predictor.load_model()
        # print(predictor.download_prediction_file())
    except Exception as e:
        raise e


