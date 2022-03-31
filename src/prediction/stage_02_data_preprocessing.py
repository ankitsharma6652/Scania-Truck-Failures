import pickle

from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
import argparse
import pandas as pd
import os,sys
from sklearn.impute import SimpleImputer, MissingIndicator
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn import over_sampling
from sklearn.decomposition import PCA
import pickle as p
from src.utils.DbOperations_Logs import DBOperations
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from src.exception import CustomException

class preprocessing:

    """ This class is for doing preprocessing on dataset"""

    def __init__(self, config_path, params_path):
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
        self.target_column = self.params["target_columns"]['columns']
        # self.df = df.drop(columns=self.target_column, inplace=True)
        self.standard_scale_file_name=self.params['preprocesssing_objects']['standard_scale_file_name']
        self.pca_file_name=self.params['preprocesssing_objects']['pca_file_name']
        self.label_encoding_file_name=self.params['preprocesssing_objects']['label_encoding_file_name']
        self.imputer_file_name=self.params['preprocesssing_objects']['imputer_file_name']
        access_key, secret_access_key = self.db_logs.get_aws_s3_keys()
        self.aws = AmazonSimpleStorageService(access_key, secret_access_key, self.config['storage']['bucket_name'])
    def get_label_column(self, df, label):
        try:
            self.target_column = df[label]
            self.df.drop(columns=['class'], inplace=True)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_label_column",
                                     f"Seperated the label column from Dataset")
            return self.target_column
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "get_label_column",
                                     f"{e}")
            raise (CustomException(e, sys)) from e
    # def get_target_column(self, config_path):
    #     self.params = read_yaml(config_path)
    #     target_column = self.params["target_columns"]['columns']
    #     return target_column

    def get_standard_scaling_object(self):
        return StandardScaler()

    def standard_scaling(self, df):
        """
        Scaling the data points between the range of 0 and 1
        :param df:  dataframe
        :return:dataframe after standard scaling and standard scaling object
        @author : Ankit Sharma
        """
        # try :
        #     self.std = self.get_standard_scaling_object()
        #     self.train_std = self.std.transform(df)
        #     self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "standard_scaling",
        #                              "Standard Scaling done on Dataset")
        #     return pd.DataFrame(self.train_std, columns=df.columns), self.std
        # except Exception as e:
        #     print(e)
        #     self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "standard_scaling",
        #                              f"{e}")
        #     return e
        print(self.standard_scale_file_name)
        try:
            # f=open(os.path.join("artifacts/preprocesssing_objects_dir",self.standard_scale_file_name),'rb')
            scaler=(self.aws.get_pickle_file("artifacts/preprocesssing_objects_dir",self.standard_scale_file_name)['file_content'])
            print(scaler)
            # print((f['file_content']))

            # scaler=(open(f['file_content'],'rb'))
            self.df=scaler.transform(df)
            print(df)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "standard_scaling",
                                         "Standard Scaling done on Dataset")
            return pd.DataFrame(self.df,columns=df.columns)

        except Exception as e:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "standard_scaling",
                                         e)
            raise (CustomException(e, sys)) from e


    def remove_missing_values_columns(self, df):
     
        try:
            self.df = df
            self.df.replace(to_replace=['na', 'nan'], value=np.NaN, inplace=True)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
                                     "remove_missing_values_columns",
                                     "Replaced the na and nan values with np.Nan")
            self.df.dropna(axis=1, thresh=self.df.shape[0] * 0.7, inplace=True)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
                                     "remove_missing_values_columns",
                                     "Removed the columns where >=70 % values are missing")
            # print(df.columns.shape)
            return self.df
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
                                     "remove_missing_values_columns",
                                     f"{e}")
            raise (CustomException(e, sys)) from e

    def handle_missing_values_using_median_imputation(self, df):

        try:
            self.df = df
            # self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
            #                          "handle_missing_values_using_median_imputation",
            #                          "")
            # f=open(os.path.join("artifacts/preprocesssing_objects_dir",self.imputer_file_name),'rb')
            # self.impute_median=pickle.load(open(os.path.join("artifacts/preprocesssing_objects_dir",self.imputer_file_name),'rb'))
            print((self.aws.get_pickle_file("artifacts/preprocesssing_objects_dir",self.imputer_file_name)['status']))
            self.impute_median=(self.aws.get_pickle_file("artifacts/preprocesssing_objects_dir",self.imputer_file_name)['file_content']) # Reading median imputer file from AWS S3 storage

            self.df=self.impute_median.transform(df)
            # self.impute_median = SimpleImputer(missing_values = np.nan, strategy='median', copy=True, verbose=2)
            # self.df_imputed_median = pd.DataFrame(self.impute_median.transform(self.df), columns=self.df.columns)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
                                     "handle_missing_values_using_median_imputation",
                                     f"Missing Values handled by the Median imputation technique")
            # print(self.df_imputed_median)
            # print(df_imputed_median.isna().sum().sum())
            # print(df_imputed_median.shape)
            # df_imputed_median['class']=df['class']
            print("Inside handle_missing_values_using_median_imputation()")
            print(pd.DataFrame(self.df,columns=df.columns))
            return pd.DataFrame(self.df,columns=df.columns)
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "handle_missing_values_using_median_imputation",
                                     f"{e}")
            raise (CustomException(e, sys)) from e

    # def standard_scaling(self, df):

    #     try :
    #         std_scaling = os.path.join("artifacts_dir", "standard_scalar", "standard_scale_file_path", "standard_scale_file_name" )
    #         scaler = p.load(open(std_scaling, 'rb'))
    #         df_scaled = scaler.transform(df)
    #         self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "standard_scaling",
    #                                  "Standard Scaling done on  Test Dataset")
    #         return pd.DataFrame(df_scaled, columns= df.columns)

    #     except Exception as e:
    #         print(e)
    #         self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "standard_scaling",
    #                                  f"{e}")
    #         return e

    def dimensionality_reduction_using_pca(self, df, n_components, random_state):
        """
        Performing dimensionality reduction using pca
        selecting 90 features(components) as they are  explaining 97% of data

        """
        try:
            
            self.df = df
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "dimensionality_reduction_using_pca",
                                     "Performing dimensionality reduction using pca selecting 90 features(components) as they are  explaining 97% of data")
            # self.df_pca = PCA(n_components=n_components, random_state=random_state)
            self.df_pca=(self.aws.get_pickle_file("artifacts/preprocesssing_objects_dir",self.pca_file_name)['file_content']) # Reading median PCA object file from AWS S3 storage

            # self.df = self.df_pca.fit_transform(self.df)
            # self.df_pca=pickle.load(open(os.path.join("artifacts/preprocesssing_objects_dir",self.pca_file_name),'rb'))
            self.df=self.df_pca.transform(self.df)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "dimensionality_reduction_using_pca",
                                     "Completed dimensionality_reduction using PCA done on Test Dataset")
            return pd.DataFrame(self.df)
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "dimensionality_reduction_using_pca",
                                     f"{e}")
            raise (CustomException(e, sys)) from e

    def label_encoding(self, df):
        """encode labels to 0 and 1"""
        try:
            self.df = df
            # self.le = LabelEncoder()
            # df = self.get_target_column(config_path)
            # self.df['class'] = self.le.fit_transform(self.df['class'])

            # self.df = self.df.copy()
            # self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "label_encoding",
            #                          "Converted the categorical values from label column to 0 and 1")
            # return self.df
            # self.le=pickle.load(open(os.path.join("artifacts/preprocesssing_objects_dir",self.label_encoding_file_name),'rb'))
            self.le=(self.aws.get_pickle_file("artifacts/preprocesssing_objects_dir",self.label_encoding_file_name)['file_content']) # Reading median PCA object file from AWS S3 storage

            self.df['class'] = self.le.transform(self.df['class'])
            # self.df=self.le.transform(self.df)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "label_encoding",
                                     "Converted the categorical values from label column to 0 and 1")
            return self.df
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "label_encoding",
                                     f"{e}")
            raise (CustomException(e, sys)) from e

    def remove_highly_corr_features(self, df):
        """
            Setting correlation coefficient threshold as 0.8 to remove highly correlated features in train data and
            Removing Highly correlated features
            @author : Ankit Sharma
        """
        try:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "remove_highly_corr_features",
                                     "Setting correlation coefficient threshold as 0.8 to remove highly correlated features in train data and Removing Highly correlated features")
            self.df = df
            self.corr_matrix = self.df.corr()
            self.mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))
            self.tri_df = self.corr_matrix.mask(self.mask)
            self.to_drop = [c for c in self.tri_df.columns if any(self.tri_df[c] > 0.8)]
            # self.db_logs.insert_logs(self.training_table_name, self.stage_name, "remove_highly_corr_features",
            #                          f"Highly Correlated Features - {str([self.to_drop])}")
            self.df_imp_features = self.df.drop(self.df[self.to_drop], axis=1)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "remove_highly_corr_features",
                                     "Highly Correlated Features Removed from the dataset")
            return self.df_imp_features

        except Exception as e:
            # print(e)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "remove_highly_corr_features",
                                     f"{e}")

            raise (CustomException(e, sys)) from e

    def upsampling_postive_class(self, df):
        """
        upsampling the positive class using smote technique to have balanced dataset.
        @author : Ankit Sharma
        """
        try:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "upsampling_postive_class",
                                     f"upsampling the positive class using smote technique to have balanced dataset.")
            self.df = df
            # Upsampling the positive class using Smote Technique
            df.sm = over_sampling.SMOTE()
            # sm = over_sampling.SMOTE()
            self.df_Sampled_Smote, self.y_train = self.df.sm.fit_resample(self.df, self.df['class'])
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "upsampling_postive_class",
                                     "upsampled the positive class.")
            return self.df_Sampled_Smote
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "upsampling_postive_class",
                                     f"{e}")
            raise (CustomException(e, sys)) from e

    def downsampling_neg_class(self, df):
        """
        #downsampling the negative class using smote technique to have balanced dataset
        @author : Ankit Sharma
        """
        try:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "downsampling_neg_class",
                                     f"downsampling the negative class using smote technique to have balanced dataset")
            self.df = df
            self.df_target = self.df['class']
            self.train_neg_sampled = self.df[self.df_target == 0].sample(n=10000, random_state=42)
            self.train_Sampled = self.df[self.df_target == 1].append(self.train_neg_sampled)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "downsampling_neg_class",
                                     f"downsampled the negative class ")
            return self.train_Sampled
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "downsampling_neg_class",
                                     f"{e}")
            raise (CustomException(e, sys)) from e

    def data_preprocessing(self):

        try:
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
                                     "data_preprocessing",
                                     "Data Preprocessing on test data Started")
            # config = read_yaml(config_path)
            # params = read_yaml(params_path)
            n_components = self.params['base']['n_components']
            random_state = self.params['base']['random_state']

            artifacts_dir = self.config["artifacts"]['artifacts_dir']
            local_data_dirs = self.config["artifacts"]['local_data_dirs']
            local_data_test_file = self.config["artifacts"]['local_data_test_file']
            label = self.params["target_columns"]['columns']
            standard_scaling_file_dir = self.params['standard_scalar']['standard_scale_file_path']
            standard_scale_predfile_name = self.params['standard_scalar']['standard_scale_predfile_name']
            # raw_local_file_path = os.path.join(artifacts_dir, local_data_dirs, local_data_test_file)
            #
            # print(raw_local_file_path)
            raw_test_file = (os.path.join(artifacts_dir, local_data_dirs)).replace('\\', '/')
            # self.df = pd.read_csv(raw_local_file_path)
            # print(raw_local_file_path)
            # self.df = pd.read_csv(raw_local_file_path) # Local File
            print((raw_test_file, local_data_test_file))
            self.df = self.aws.read_csv_file(raw_test_file, local_data_test_file)[
                'data_frame']  # Reading Data from S3 Storage
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "data_preprocessing",
                                     "Successfully  read the test dataset from s3 Storage")

            # print(self.df)

            # print(self.df)
            self.df_after_removing_missing_values_columns = self.remove_missing_values_columns(self.df)
            print("After remove_missing_values_columns",self.df_after_removing_missing_values_columns)
            self.df_after_label_encoding = self.label_encoding(self.df_after_removing_missing_values_columns)
            print("After Label encoding",self.df_after_label_encoding)
            # print(self.df_after_label_encoding.isna().sum())
            self.df_missing_values_handled = self.handle_missing_values_using_median_imputation(self.df_after_label_encoding)
            print("After Missing values handled",self.df_missing_values_handled)
            # print(self.df_missing_values_handled.isna().sum())
            # self.handle_missing_values_using_median_imputation(self.df_after_label_encoding)[0]
            # df = self.handle_missing_values_using_median_imputation(df)
            # self.df_remove_highly_correlated_features = self.remove_highly_corr_features(self.df_missing_values_handled)
            self.df_upsampled_pos_class = self.upsampling_postive_class(
                self.downsampling_neg_class(self.df_missing_values_handled))
            self.target_column = self.get_label_column(self.df_upsampled_pos_class, label)
            # df = self.standard_scaling(self.df_upsampled_pos_class)
            self.df_after_pca = self.dimensionality_reduction_using_pca(self.df_upsampled_pos_class, n_components,
                                                                        random_state)
            self.standard_scalar_data = self.standard_scaling(self.df_after_pca)
            print("Standard scaling completed")
            print(self.standard_scalar_data)

            preprocessed_data_dir = self.config["artifacts"]["preprocessed_data_dir"]
            target_column_data_dir = self.config['artifacts']['target_column_data_dir']

            create_directory_path([os.path.join(artifacts_dir, preprocessed_data_dir)])
            create_directory_path([os.path.join(artifacts_dir, target_column_data_dir)])
            create_directory_path([os.path.join(artifacts_dir, standard_scaling_file_dir)])

            preprocessed_test_file = self.config["artifacts"]["preprocessed_test_file"]
            target_column_testdata_file = self.config["artifacts"]["target_column_testdata_file"]
            # target_column_data_dir: target_column_data_dir
            # target_column_data_file: target_column_testing_data
            preprocessed_data_path = os.path.join(artifacts_dir, preprocessed_data_dir, preprocessed_test_file)
            target_column_data_path = os.path.join(artifacts_dir, target_column_data_dir, target_column_testdata_file)
            # # standard_scaling_data_path = os.path.join(artifacts_dir, standard_scaling_file_dir,
            #                                           standard_scale_predfile_name)
            # # save_local_df(self.standard_scalar_data, preprocessed_data_path)

            save_local_df(self.target_column, target_column_data_path)
            save_local_df(self.standard_scalar_data, preprocessed_data_path)
            self.aws.upload_file(os.path.join(artifacts_dir, preprocessed_data_dir).replace('\\', '/'),
                                 preprocessed_test_file, preprocessed_test_file, local_file_path=preprocessed_data_path,
                                 over_write=True)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "data_preprocessing",
                                     "Preprocessed test data file uploaded to  s3 storage")
            # save_local_df(self.standard_scalar_data,preprocessed_data_path)
            self.aws.upload_file(os.path.join(artifacts_dir, target_column_data_dir).replace('\\', '/'),
                                 target_column_testdata_file, target_column_testdata_file, local_file_path=target_column_data_path,
                                 over_write=True)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name, "data_preprocessing",
                                     "Preprocessed target data file uploaded to  s3 storage")
            # with open(standard_scaling_data_path, 'wb') as s:
            #     p.dump(self.standard_scaling_object, s)
            # self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
            #                          "data_preprocessing",
            #                          f"Standard Scaler object file saved at {standard_scaling_data_path}")
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
                                     "data_preprocessing",
                                     f"Data Preprocessing on Test dataset file saved at : {preprocessed_data_path}")
            # self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
            #                          "data_preprocessing",
            #                          f"Label Column file saved  at : {target_column_data_path}")
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
                                     "data_preprocessing",
                                     "Data Preprocessing Completed")
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.prediction_table_name, self.stage_name,
                                     "data_preprocessing",
                                     f"{e}")
            raise (CustomException(e, sys)) from e


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--params", default="config/params.yaml")

    parsed_args = args.parse_args()

    try:

        preprocessing_object = preprocessing(config_path=parsed_args.config, params_path=parsed_args.params)
        preprocessing_object.data_preprocessing()


    except Exception as e:
        raise (CustomException(e, sys)) from e
