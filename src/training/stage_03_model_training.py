import time
from src.utils.email_sender.email_sender import email_sender
from sklearn.ensemble import RandomForestClassifier
import shutil
from src.utils.all_utils import read_yaml, create_directory_path, save_local_df,model_training_logs
import argparse
import pandas as pd
import os,sys
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer,MissingIndicator
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from imblearn import over_sampling
from sklearn.decomposition import PCA
import pickle as p
from sklearn.metrics import confusion_matrix,f1_score,log_loss,roc_curve,recall_score,precision_recall_curve,precision_score,fbeta_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from src.utils.DbOperations_Logs import DBOperations
from sklearn.metrics  import roc_auc_score,accuracy_score
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService

class ModelTraining:
    def __init__(self,config_path,params_path,model_path,recievers_mail:str=None):
        # self.file_object = file_object
        # self.logger_object = logger_object
        self.email_sender=email_sender()
        self.TO=recievers_mail
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier()
        self.stage_name= os.path.basename(__file__)[:-3]
        # self.recievers_email=recievers_email
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.database_name = self.params['logs_database']['database_name']
        self.training_table_name = self.params['logs_database']['training_table_name']
        self.model_training_thread_table_name=self.params['model_training_thread']['model_training_thread_table_name']
        self.user_name = self.config['database']['user_name']
        self.password = self.config['database']['password']
        self.db_logs = DBOperations(self.database_name)
        self.db_logs.establish_connection(self.user_name, self.password)
        self.db_logs.best_model_table()
        self.model=read_yaml(model_path)
        self.split_ratio = self.params["base"]["test_size"]
        self.random_state = self.params["base"]["random_state"]
        self.artifacts_dir = self.config["artifacts"]['artifacts_dir']
        self.standard_scaling_file_dir = self.params['standard_scalar']['standard_scale_file_path']
        self.standard_scale_file_name = self.params['standard_scalar']['standard_scale_file_name']
        self.standard_scaling_data_path = os.path.join(self.artifacts_dir, self.standard_scaling_file_dir, self.standard_scale_file_name)
        self.preprocessed_data_dir = self.config["artifacts"]["preprocessed_data_dir"]
        self.target_column_data_dir = self.config['artifacts']['target_column_data_dir']
        self.preprocessed_data_file = self.config["artifacts"]["preprocessed_data_file"]
        self.target_column_data_file = self.config["artifacts"]["target_column_data_file"]
        self.model_dir=self.model['model']['model_dir']
        self.Random_forest = self.model['model']['random_forest']
        self.Xgboost=self.model['model']['xgboost']
        access_key, secret_access_key = self.db_logs.get_aws_s3_keys()
        self.aws = AmazonSimpleStorageService(access_key, secret_access_key, self.config['storage']['bucket_name'])
        self.preprocessed_data_path = os.path.join(self.artifacts_dir, self.preprocessed_data_dir, self.preprocessed_data_file)
        self.target_column_data_path = os.path.join(self.artifacts_dir, self.target_column_data_dir, self.target_column_data_file)
        self.db_logs.model_training_thread(self.model_training_thread_table_name)
    def get_best_params_for_xgboost(self, train_x, train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: Ankit Sharma
                                        Version: 1.0
                                        Revisions: None

                                """
        # self.logger_object.log(self.file_object,
        #                        'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                # 'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10],
                'n_estimators': [100,200,500,1000,2000]

            }
            # Creating an object of the Grid Search class
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_params_for_xgboost",
                                     f" Best parameters finding started using Grid Search CV")
            self.grid = GridSearchCV(XGBClassifier(use_label_encoder=False), self.param_grid_xgboost, verbose=10,
                                     cv=5,scoring='f1')
            # finding the best parameters
            self.grid.fit(train_x, train_y)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_params_for_xgboost",
                                     f" Best parameters finding  Ended")

            # extracting the best parameters
            # self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_params_for_xgboost", f"Got the Best Parameters for XgBoost:{self.max_depth},{self.n_estimators}")
            # creating a new model with the best parameters
            self.xgb = XGBClassifier( max_depth=self.max_depth,
                                     n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            # self.logger_object.log(self.file_object,
            #                        'XGBoost best params: ' + str(
            #                            self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            # self.logger_object.log(self.file_object,
            #                        'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
            #                            e))
            # self.logger_object.log(self.file_object,
            #                        'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_params_for_xgboost",
                                     f"{e}")
            raise (CustomException(e, sys)) from e

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Ankit Sharma
                                Version: 1.0
                                Revisions: None

                        """

        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [100,200,500,1000,2000],"max_depth": [3,5,10]}
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_params_for_random_forest",
                                     f" Best parameters finding started using Grid Search CV")
            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=10,scoring='f1')
            #finding the best parameters
            self.grid.fit(train_x, train_y)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_params_for_random_forest",
                                     f" Best parameters finding  Ended")
            #extracting the best parameters
            # self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            # self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_params_for_random_forest",
                                     f"Got the Best Parameters for Random Forest:{self.max_depth},{self.n_estimators}")
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                          max_depth=self.max_depth)
            # training the mew model
            self.clf.fit(train_x, train_y)

            return self.clf
        except Exception as e:
            # self.logger_object.log(self.file_object,
            #                    'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
            #                        e))
            # self.logger_object.log(self.file_object,
            #                    'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_params_for_random_forest",
                                    f"{e}")
            raise (CustomException(e, sys)) from e
    def get_total_cost(self,con_mat):
        print("-" * 117)
        print('Confusion Matrix: ', '\n', con_mat)
        print("-" * 117)
        print("Type 1 error (False Positive) = ", con_mat[0][1])
        print("Type 2 error (False Negative) = ", con_mat[1][0])
        print("-" * 117)
        print("Total cost = ", con_mat[0][1] * 10 + con_mat[1][0] * 500)
        print("-" * 117)
        return {"Type 1 error (False Positive)": con_mat[0][1], "Type 2 error (False Negative)":con_mat[1][0],"Total cost":con_mat[0][1] * 10 + con_mat[1][0] * 500}

    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the less cost.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: Ankit Sharma
                                                Version: 1.0
                                                Revisions: None

                                        """
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model
            self.y_pred_prob_xgboost= self.xgboost.predict_proba(test_x)[:, 1] > 0.2   # At 0.2, we observe that precision is almost more than 95% and recall is almost around 98%. We want our recall to be near to 100% and at the same time we also want our precision to be high.
            self.con_mat_xgboost = confusion_matrix(test_y, self.y_pred_prob_xgboost)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_model",
                                     f"XgBoost:{self.con_mat_xgboost}")
            self.total_cost_xgboost_model=self.get_total_cost(self.con_mat_xgboost)['Total cost']
            # create best model for Random Forest
            self.random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            self.prediction_random_forest = self.random_forest.predict(test_x)# prediction using the Random Forest Algorithm
            self.y_pred_prob_random_forest=self.random_forest.predict_proba(test_x)[:,1]  > 0.30 # At 0.3, we observe that precision is almost more than 95% and recall is almost around 98%. We want our recall to be near to 100% and at the same time we also want our precision to be high.
            self.con_mat_random_forest=confusion_matrix(test_y, self.y_pred_prob_random_forest)
            self.total_cost_random_forest=self.get_total_cost(self.con_mat_random_forest)['Total cost']
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_model",
                                     f"Random Forest:{self.con_mat_random_forest}")
            if self.total_cost_random_forest > self.total_cost_xgboost_model :
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_model",
                                        "Best Model is Xg-Boost")

                return self.Xgboost,self.xgboost,self.total_cost_xgboost_model
            else:
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_model",
                                         "Best Model is Random Forest")
                return self.Random_forest, self.random_forest,self.total_cost_random_forest
        except Exception as e:
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_best_model",
                                     f"{e}")
            print(e)
            raise (CustomException(e, sys)) from e


    def empty_model_dir(self):
        """
        This class shall be used to empty the model dir
        """
        try:
            shutil.rmtree(os.path.join(self.artifacts_dir, self.model_dir))
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "empty_model_dir",
                             f"Deleted the old models in {os.path.join(self.artifacts_dir, self.model_dir)}")
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "empty_model_dir",
                                     f"{e}")
            raise (CustomException(e, sys)) from e

    def start_model_training(self):
        try:
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "start_model_training",
                                     "Model Training process Started")
            time.sleep(1)
            # self.empty_model_dir()
            create_directory_path([os.path.join(self.artifacts_dir, self.model_dir)])

            # self.training_data=pd.read_csv(self.preprocessed_data_path)
            # self.target_column_data=pd.read_csv(self.target_column_data_path).iloc[:,0]
            # print(type(self.target_column_data.iloc[0]))
            self.training_data=self.aws.read_csv_file(os.path.join(self.artifacts_dir, self.preprocessed_data_dir).replace('\\','/'), self.preprocessed_data_file)['data_frame']
            self.target_column_data = self.aws.read_csv_file(os.path.join(self.artifacts_dir, self.target_column_data_dir).replace('\\','/'), self.target_column_data_file)['data_frame'].iloc[:, 0]
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.training_data,self.target_column_data,test_size=self.split_ratio, random_state=self.random_state)
            print(f"x train shape:{self.x_train.shape}")
            print(f"y train shape:{self.y_train.shape}")
            print(f"x test shape:{self.x_test.shape}")
            print(f"y test shape:{self.y_test.shape}")
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "start_model_training",
                                     f"Splitted the data->  x train shape:{self.x_train.shape} , y train shape:{self.y_train.shape}, x test shape:{self.x_test.shape} , y test shape:{self.y_test.shape}")
            # f=open(self.standard_scaling_data_path,'rb')
            # scaling_object=p.load(f)

            # self.get_best_params_for_random_forest(self.x_train,self.y_train)
            # self.get_best_params_for_xgboost(self.x_train,self.y_train)
            # with open(self.standard_scaling_data_path,'rb') as std:
            #     scaling_object = p.load(std)
            self.best_model_name, self.best_model,self.total_cost=self.get_best_model(self.x_train,self.y_train,(self.x_test),self.y_test)
            self.model_dir_path = os.path.join(self.artifacts_dir, self.model_dir,f"{self.best_model_name}.pkl")
            self.mail_text=f"""
            Congratulations, Model Training  has completed.
            Please find the below details about the Trained Model-
            Model Name:{self.best_model_name}
            Model Saved at : {self.model_dir_path},
            Total Cost Obtained : {self.total_cost}
            
            Please perform prediction.
            """
            if self.db_logs.get_best_model_name() is None:
                self.db_logs.insert_best_model(self.best_model_name)
            else:
                self.db_logs.update_best_model_name(best_model_name=self.best_model_name)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "start_model_training",
                                     f"Best Model Saved at : {self.model_dir_path}")
            with open(self.model_dir_path,'wb') as model_file:
                p.dump(self.best_model,model_file)
            self.aws.upload_file(os.path.join(self.artifacts_dir, self.model_dir).replace("\\",'/'),f"{self.best_model_name}.pkl",f"{self.best_model_name}.pkl",local_file_path=os.path.join(self.artifacts_dir, self.model_dir,f"{self.best_model_name}.pkl"),over_write=True)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "start_model_training",
                                     f"Model training file uploaded to the S3 at location {self.model_dir_path}")
            self.db_logs.update_model_training_thread_status('C')
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "start_model_training",
                                     "Model Training process ended")
            if self.TO is not None:

                self.email_sender.send_email(mail_text=self.mail_text,TO=self.TO)
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "start_model_training",
                                        f"Model Training complete Notification has been sent to {self.TO} with the content {self.mail_text}")
                print("email sent",self.TO)
        except Exception as e:
            print(e)
            self.db_logs.update_model_training_thread_status('NS')
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "start_model_training",
                                     f"{e}")
            print('e')
            raise (CustomException(e, sys)) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="config/params.yaml")
    args.add_argument("--model", "-m", default="config/model.yaml")

    parsed_args = args.parse_args()
    model_training=ModelTraining(config_path=parsed_args.config, params_path=parsed_args.params,model_path=parsed_args.model)
    model_training.start_model_training()
