from cProfile import label
from signal import default_int_handler
from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, MissingIndicator
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler,Normalizer
from imblearn import over_sampling
from sklearn.decomposition import PCA
import logging
import pickle as p
from src.utils.DbOperations_Logs import DBOperations
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from pathlib import  Path
class preprocessing:
    """ This class is for doing preprocessing on dataset"""

    def __init__(self, config_path, params_path):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.stage_name = os.path.basename(__file__)[:-3]
        self.database_name = self.params['logs_database']['database_name']
        self.training_table_name = self.params['logs_database']['training_table_name']
        self.user_name = self.config['database']['user_name']
        self.password = self.config['database']['password']
        self.db_logs = DBOperations(self.database_name)
        self.db_logs.establish_connection(self.user_name, self.password)
        self.db_logs.create_table(self.training_table_name)
        # self.target_column = self.params["target_columns"]['columns']
        # self.df = df.drop(columns=self.target_column, inplace=True)
        access_key, secret_access_key = self.db_logs.get_aws_s3_keys()
        self.aws = AmazonSimpleStorageService(access_key, secret_access_key, self.config['storage']['bucket_name'])
    def get_label_column(self, df, label):
        try:
            self.target_column = df[label]
            self.df.drop(columns=['class'], inplace=True)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_label_column",
                                     f"Seperated the label column from Dataset")
            return self.target_column
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "get_label_column",
                                     f"{e}")
            return(e)

    # def get_target_column(self, config_path):
    #     self.params = read_yaml(config_path)
    #     target_column = self.params["target_columns"]['columns']
    #     return target_column

    def get_standard_scaling_object(self):
        # return StandardScaler()
        return Normalizer()

    def standard_scaling(self, df):
        """
        Scaling the data points between the range of 0 and 1
        :param df:  dataframe
        :return:dataframe after standard scaling and standard scaling object
        @author : Ankit Sharma
        """
        try :
            self.std = self.get_standard_scaling_object()
            self.train_std = self.std.fit_transform(df)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "standard_scaling",
                                     "Standard Scaling done on Dataset")
            return pd.DataFrame(self.train_std, columns=df.columns), self.std
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "standard_scaling",
                                     f"{e}")
            return e

    def dimensionality_reduction_using_pca(self, df, n_components, random_state):
        """
        Performing dimensionality reduction using pca
        selecting 90 features(components) as they are  explaining 97% of data
            @author : Ankit Sharma
        """
        try:
            self.df = df
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "dimensionality_reduction_using_pca",
                                     "Performing dimensionality reduction using pca selecting 90 features(components) as they are  explaining 97% of data")
            self.df_pca = PCA(n_components=n_components, random_state=random_state)

            self.df = self.df_pca.fit_transform(self.df)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "dimensionality_reduction_using_pca",
                                     "Completed dimensionality_reduction using PCA done on Dataset")
            return pd.DataFrame(self.df),self.df_pca
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "dimensionality_reduction_using_pca",
                                     f"{e}")
            return e

    def label_encoding(self, df):
        """encode labels to 0 and 1"""
        try:
            self.df = df
            self.le = LabelEncoder()
            # df = self.get_target_column(config_path)
            self.df['class'] = self.le.fit_transform(self.df['class'])

            self.df = self.df.copy()
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "label_encoding",
                                     "Converted the categorical values from label column to 0 and 1")
            return self.df,self.le
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "label_encoding",
                                     f"{e}")
            return e

    def remove_highly_corr_features(self, df):
        """
            Setting correlation coefficient threshold as 0.8 to remove highly correlated features in train data and
            Removing Highly correlated features
            @author : Ankit Sharma
        """
        try:
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "remove_highly_corr_features",
                                     "Setting correlation coefficient threshold as 0.8 to remove highly correlated features in train data and Removing Highly correlated features")
            self.df = df
            self.corr_matrix = self.df.corr()
            self.mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))
            self.tri_df = self.corr_matrix.mask(self.mask)
            self.to_drop = [c for c in self.tri_df.columns if any(self.tri_df[c] > 0.8)]
            # self.db_logs.insert_logs(self.training_table_name, self.stage_name, "remove_highly_corr_features",
            #                          f"Highly Correlated Features - {str([self.to_drop])}")
            self.df_imp_features = self.df.drop(self.df[self.to_drop], axis=1)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "remove_highly_corr_features",
                                     "Highly Correlated Features Removed from the dataset")
            return self.df_imp_features
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "remove_highly_corr_features",
                                     f"{e}")

            return e

    def upsampling_postive_class(self, df):
        """
        upsampling the positive class using smote technique to have balanced dataset.
        @author : Ankit Sharma
        """
        try:
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "upsampling_postive_class",
                                     f"upsampling the positive class using smote technique to have balanced dataset.")
            self.df = df
            # Upsampling the positive class using Smote Technique
            df.sm = over_sampling.SMOTE()
            # sm = over_sampling.SMOTE()
            self.df_Sampled_Smote, self.y_train = self.df.sm.fit_resample(self.df, self.df['class'])
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "upsampling_postive_class",
                                     "upsampled the positive class.")
            return self.df_Sampled_Smote
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "upsampling_postive_class",
                                     f"{e}")
            return e
    def downsampling_neg_class(self, df):
        """
        #downsampling the negative class using smote technique to have balanced dataset
        @author : Ankit Sharma
        """
        try:
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "downsampling_neg_class",
                                     f"downsampling the negative class using smote technique to have balanced dataset")
            self.df = df
            self.df_target = self.df['class']
            self.train_neg_sampled = self.df[self.df_target == 0].sample(n=10000, random_state=42)
            self.train_Sampled = self.df[self.df_target == 1].append(self.train_neg_sampled)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "downsampling_neg_class",
                                     f"downsampled the negative class ")
            return self.train_Sampled
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "downsampling_neg_class",
                                     f"{e}")
            return e
    def handle_missing_values_using_median_imputation(self, df):
        """
        fill the missing values by using Median Imputation .
        :return:imputed dataframe, imputed object
        @author : Ankit Sharma
        """
        try:
            self.df = df
            # self.db_logs.insert_logs(self.training_table_name, self.stage_name,
            #                          "handle_missing_values_using_median_imputation",
            #                          f"{e}")
            # print(self.df)
            # df1=df.drop(columns=['class'])
            self.impute_median = SimpleImputer(missing_values=np.nan, strategy='median', copy=True, verbose=2)
            self.df_imputed_median = pd.DataFrame(self.impute_median.fit_transform(self.df), columns=self.df.columns)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name,
                                     "handle_missing_values_using_median_imputation",
                                     f"Missing Values handled by the Median imputation technique")
            # print(self.df_imputed_median)
            # print(df_imputed_median.isna().sum().sum())
            # print(df_imputed_median.shape)
            # df_imputed_median['class']=df['class']
            return self.df_imputed_median, self.impute_median
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "handle_missing_values_using_median_imputation",
                                     f"{e}")
            return e

    def remove_missing_values_columns(self, df):
        """
        Replace the na and nan values with np.Nan and Removes the columns where >=70 % values are missing,
        @author : Ankit Sharma
        """
        try:
            self.df = df
            self.df.replace(to_replace=['na', 'nan'], value=np.NaN, inplace=True)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name,
                                     "remove_missing_values_columns",
                                     "Replaced the na and nan values with np.Nan")
            self.df.dropna(axis=1, thresh=self.df.shape[0] * 0.7, inplace=True)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name,
                                     "remove_missing_values_columns",
                                     "Removed the columns where >=70 % values are missing")
            # print(df.columns.shape)
            return self.df
        except Exception as e:
            print(e)
            self.db_logs.insert_logs(self.training_table_name, self.stage_name,
                                     "remove_missing_values_columns",
                                     f"{e}")
            return  e

    def data_preprocessing(self):

        try:
        # config = read_yaml(config_path)
        # params = read_yaml(params_path)
            n_components = self.params['base']['n_components']
            random_state = self.params['base']['random_state']

            artifacts_dir = self.config["artifacts"]['artifacts_dir']
            local_data_dirs = self.config["artifacts"]['local_data_dirs']
            local_data_train_file = self.config["artifacts"]['local_data_train_file']
            label = self.params["target_columns"]['columns']
            # standard_scaling_file_dir=self.params['standard_scalar']['standard_scale_file_path']
            # standard_scale_file_name=self.params['standard_scalar']['standard_scale_file_name']
            preprocesssing_objects_file_dir = self.params['preprocesssing_objects']['preprocesssing_objects_path']
            standard_scale_file_name = self.params['preprocesssing_objects']['standard_scale_file_name']
            label_encoding_file_name=self.params['preprocesssing_objects']['label_encoding_file_name']
            imputer_file_name=self.params['preprocesssing_objects']['imputer_file_name']
            pca_file_name=self.params['preprocesssing_objects']['pca_file_name']
            raw_local_file_path = os.path.join(artifacts_dir, local_data_dirs, local_data_train_file)
            raw_training_file=(os.path.join(artifacts_dir,local_data_dirs)).replace('\\','/')
            # print(raw_local_file_path)
            # self.df = pd.read_csv(raw_local_file_path) # Local File
            print((raw_training_file,local_data_train_file))
            self.df = self.aws.read_csv_file(raw_training_file,local_data_train_file)['data_frame'] # Reading Data from S3 Storage
            self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                 "Successfully the read the training dataset from s3 Storage")
            print(self.df)
            self.df_after_removing_missing_values_columns=self.remove_missing_values_columns(self.df)
            # print(self.df_after_removing_missing_values_columns)
            self.df_after_label_encoding,self.label_encoding_object = self.label_encoding(self.df_after_removing_missing_values_columns)
            # print(self.df_after_label_encoding)
            self.df_missing_values_handled,self.imputer_object = self.handle_missing_values_using_median_imputation(self.df_after_label_encoding)
            # df = self.handle_missing_values_using_median_imputation(df)
            # self.df_remove_highly_correlated_features = self.remove_highly_corr_featues(self.df_missing_values_handled)
            self.df_upsampled_pos_class = self.upsampling_postive_class(self.downsampling_neg_class(self.df_missing_values_handled))
            self.target_column = self.get_label_column(self.df_upsampled_pos_class, label)
            # df = self.standard_scaling(self.df_upsampled_pos_class)
            self.df_after_pca,self.pca_object = self.dimensionality_reduction_using_pca(self.df_upsampled_pos_class, n_components, random_state)
            self.standard_scalar_data,self.standard_scaling_object=self.standard_scaling(self.df_after_pca)
            print(self.standard_scalar_data.shape)
            # train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
            preprocessed_data_dir = self.config["artifacts"]["preprocessed_data_dir"]
            target_column_data_dir = self.config['artifacts']['target_column_data_dir']

            create_directory_path([os.path.join(artifacts_dir, preprocessed_data_dir)])
            create_directory_path([os.path.join(artifacts_dir, target_column_data_dir)])
            create_directory_path([os.path.join(artifacts_dir, preprocesssing_objects_file_dir)])

            preprocessed_data_file = self.config["artifacts"]["preprocessed_data_file"]
            target_column_data_file = self.config["artifacts"]["target_column_data_file"]
            # target_column_data_dir: target_column_data_dir
            # target_column_data_file: target_column_training_data
            preprocessed_data_path = os.path.join(artifacts_dir, preprocessed_data_dir, preprocessed_data_file)
            target_column_data_path = os.path.join(artifacts_dir, target_column_data_dir, target_column_data_file)
            standard_scaling_data_path=os.path.join(artifacts_dir,preprocesssing_objects_file_dir,standard_scale_file_name)
            label_encoding_data_path = os.path.join(artifacts_dir, preprocesssing_objects_file_dir,
                                                  label_encoding_file_name)
            pca_data_path = os.path.join(artifacts_dir, preprocesssing_objects_file_dir,
                                               pca_file_name)
            imputer_data_path = os.path.join(artifacts_dir, preprocesssing_objects_file_dir,
                                     imputer_file_name)
            save_local_df(self.standard_scalar_data, preprocessed_data_path)
            if not self.aws.is_file_present(os.path.join(artifacts_dir, preprocessed_data_dir).replace('\\','/'),preprocessed_data_file)['status']:
                "Uploading the preprocessed data to the S3 Storage if not exists"
                self.aws.upload_file(os.path.join(artifacts_dir, preprocessed_data_dir).replace('\\','/'),preprocessed_data_file,preprocessed_data_file,local_file_path=preprocessed_data_path,over_write=True)
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                        "Preprocessed data already exists in s3 storage")
            else:
                self.db_logs.insert_logs(self.training_table_name,self.stage_name,"data_preprocessing","Uploaded the preproceesed training data to the s3 storage")
            print(os.path.join(artifacts_dir, preprocessed_data_dir))

            # self.aws.upload_file()
            save_local_df(self.target_column, target_column_data_path)
            if not self.aws.is_file_present(os.path.join(artifacts_dir, target_column_data_dir).replace('\\', '/'),
                                        target_column_data_file)['status']:
                "Uploading the Target column data to the S3 Storage if not exists"
                self.aws.upload_file(os.path.join(artifacts_dir, target_column_data_dir).replace('\\', '/'),
                                     target_column_data_file, target_column_data_file,
                                     local_file_path=target_column_data_path)
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                         "Traget Training data already exists in s3 storage")
            else:
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                         "Uploaded the Target training data to the s3 storage")
            with open(standard_scaling_data_path,'wb') as s:
                p.dump(self.standard_scaling_object,s)
            if not  self.aws.is_file_present(os.path.join(artifacts_dir, preprocesssing_objects_file_dir).replace('\\', '/'),
                                            standard_scale_file_name)['status']:
                "Uploading the standard scaling object file to S3 if not exists"
                self.aws.upload_file(os.path.join(artifacts_dir,preprocesssing_objects_file_dir).replace('\\','/'),
                                     standard_scale_file_name,standard_scale_file_name,local_file_path=standard_scaling_data_path,over_write=True)
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                    "Standard scaling object file already exists in s3 storage")
            else:
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                     "Uploaded the Standard scaling object file to the s3 storage")
            with open(label_encoding_data_path,'wb') as s:
                p.dump(self.label_encoding_object,s)
            if not self.aws.is_file_present(os.path.join(artifacts_dir, preprocesssing_objects_file_dir).replace('\\', '/'),
                                            label_encoding_file_name)['status']:
                "Uploading the Label encoding  object file to S3 if not exists"
                self.aws.upload_file(os.path.join(artifacts_dir, preprocesssing_objects_file_dir).replace('\\', '/'),
                                     label_encoding_file_name, label_encoding_file_name,
                                     local_file_path=label_encoding_data_path,over_write=True)
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                "Label encoding  object file already exists in s3 storage")
            else:
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                 "Uploaded the SLabel encoding  object file to the s3 storage")
            with open(pca_data_path,'wb') as s:
                p.dump(self.pca_object,s)
            if not self.aws.is_file_present(os.path.join(artifacts_dir, preprocesssing_objects_file_dir).replace('\\', '/'),
                                            pca_file_name)['status']:
                "Uploading the PCA  object file to S3 if not exists"
                self.aws.upload_file(os.path.join(artifacts_dir, preprocesssing_objects_file_dir).replace('\\', '/'),
                                     pca_file_name, pca_file_name,
                                     local_file_path=pca_data_path,over_write=True)
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                 "PCA object file already exists in s3 storage")
            else:
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                 "Uploaded the PCA object file to the s3 storage")
            with open(imputer_data_path,'wb') as s:
                p.dump(self.imputer_object,s)
            if  not self.aws.is_file_present(os.path.join(artifacts_dir, preprocesssing_objects_file_dir).replace('\\', '/'),
                                            pca_file_name)['status']:
                "Uploading the Imputer file  object file to S3 if not exists"
                self.aws.upload_file(os.path.join(artifacts_dir, preprocesssing_objects_file_dir).replace('\\', '/'),
                                     imputer_file_name, imputer_file_name,
                                     local_file_path=imputer_data_path,over_write=True)
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                         "Median imputation  object file already exists in s3 storage")
            else:
                self.db_logs.insert_logs(self.training_table_name, self.stage_name, "data_preprocessing",
                                         "Uploaded the Median imputation  object file to the s3 storage")

        except Exception as e:
            print(e)
            raise Exception(e)
            return e 
    

if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--params", default="config/params.yaml")

    parsed_args = args.parse_args()

    try:
        preprocessing_object = preprocessing(config_path=parsed_args.config, params_path=parsed_args.params)
        preprocessing_object.data_preprocessing()

    except Exception as e:
        raise e