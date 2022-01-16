from cProfile import label
from signal import default_int_handler
from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, MissingIndicator
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn import over_sampling
from sklearn.decomposition import PCA
import logging

logging_str = "[%(asctime)s: %(levelname)s %(module)s]: %(message)s"
log_dir = "preprocessing"

os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str,
                    filemode='a')


class preprocessing:
    """ This class is for doing preprocessing on dataset"""

    def __init__(self, config_path, params_path):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.target_column = self.params["target_columns"]['columns']
        # pass

    # def get_label_column(df,label):
    #     target_column=df[label]
    #     df.drop(columns=['class'],inplace=True)
    #     return target_column

    def get_target_column(self, config_path):
        self.params = read_yaml(config_path)
        target_column = self.params["target_columns"]['columns']
        return target_column

    def get_standard_scaling_object(self):
        return StandardScaler()

    def standard_scaling(self, df):
        """
        Scaling the data points between the range of 0 and 1
        :param df:  dataframe
        :return:dataframe after standard scaling and standard scaling object
        @author : Ankit Sharma
        """
        std = self.get_standard_scaling_object()
        train_std = std.fit_transform(df)
        return pd.DataFrame(train_std, columns=df.columns)

    def dimensionality_reduction_using_pca(self, df, n_components, random_state):
        """
        Performing dimensionality reduction using pca
        selecting 90 features(components) as they are  explaining 97% of data
            @author : Ankit Sharma
        """
        df_pca = PCA(n_components=n_components, random_state=random_state)
        df = df_pca.fit_transform(df)
        return pd.DataFrame(df)

    def label_encoding(self):
        """encode labels to 0 and 1"""

        le = LabelEncoder()
        # df = self.get_target_column(config_path)
        df_target = le.fit_transform(self.target_column)
        df = df_target.copy()
        return df

    def remove_highly_corr_featues(self, df):
        """
            Setting correlation coefficient threshold as 0.8 to remove highly correlated features in train data and
            Removing Highly correlated features
            @author : Ankit Sharma
        """
        corr_matrix = df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        tri_df = corr_matrix.mask(mask)
        to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.8)]
        df_imp_features = df.drop(df[to_drop], axis=1)
        return df_imp_features

    def upsampling_postive_class(self, df):
        """
        upsampling the positive class using smote to have balanced dataset.
        @author : Ankit Sharma
        """
        # Upsampling the positive class using Smote Technique
        sm = over_sampling.SMOTE()
        # sm = over_sampling.SMOTE()
        df_Sampled_Smote, y_train = sm.fit_resample(df, self.target_column)
        return df_Sampled_Smote

    def downsampling_neg_class(self, df):
        """
        #downsampling the negative class
        @author : Ankit Sharma
        """
        # df_target = self.get_target_column(config_path)

        train_neg_sampled = df[self.target_column == 0].sample(n=10000, random_state=42)
        train_Sampled = df[self.target_column == 1].append(train_neg_sampled)
        return train_Sampled

    def handle_missing_values_using_median_imputation(self, df):
        """
        fill the missing values by using Median Imputation .
        :return:imputed dataframe, imputed object
        @author : Ankit Sharma
        """

        # df1=df.drop(columns=['class'])
        impute_median = SimpleImputer(missing_values=np.nan, strategy='median', copy=True, verbose=2)
        df_imputed_median = pd.DataFrame(impute_median.fit_transform(df), columns=df.columns)
        # print(df_imputed_median.isna().sum().sum())
        # print(df_imputed_median.shape)
        # df_imputed_median['class']=df['class']
        return df_imputed_median, impute_median

    def remove_missing_values_columns(self, df):
        """
        Replace the na and nan values with np.Nan and Removes the columns where >=70 % values are missing,
        @author : Ankit Sharma
        """
        df.replace(to_replace=['na', 'nan'], value=np.NaN, inplace=True)
        df.dropna(axis=1, thresh=df.shape[0] * 0.7, inplace=True)
        # print(df.columns.shape)
        return df

    def data_preprocessing(self):
        # config = read_yaml(config_path)
        # params = read_yaml(params_path)
        n_components = self.params['base']['n_components']
        random_state = self.params['base']['random_state']

        artifacts_dir = self.config["artifacts"]['artifacts_dir']
        local_data_dirs = self.config["artifacts"]['local_data_dirs']
        local_data_train_file = self.config["artifacts"]['local_data_train_file']
        # label = params["target_columns"]['columns']
        raw_local_file_path = os.path.join(artifacts_dir, local_data_dirs, local_data_train_file)

        print(raw_local_file_path)

        df = pd.read_csv(raw_local_file_path)
        self.remove_missing_values_columns(df)
        df = self.label_encoding()
        df = self.handle_missing_values_using_median_imputation(df)[0]
        #  handle_missing_values_using_median_imputation(df)
        df = self.remove_highly_corr_featues(df)
        df = self.upsampling_postive_class(self.downsampling_neg_class(df))
        #  target_column = get_label_column(df, label)
        df = self.standard_scaling(df)
        df = self.dimensionality_reduction_using_pca(df, n_components, random_state)

        # train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
        preprocessed_data_dir = self.config["artifacts"]["preprocessed_data_dir"]
        target_column_data_dir = self.config['artifacts']['target_column_data_dir']

        create_directory_path([os.path.join(artifacts_dir, preprocessed_data_dir)])
        create_directory_path([os.path.join(artifacts_dir, target_column_data_dir)])

        preprocessed_data_file = self.config["artifacts"]["preprocessed_data_file"]
        target_column_data_file = self.config["artifacts"]["target_column_data_file"]
        # target_column_data_dir: target_column_data_dir
        # target_column_data_file: target_column_training_data
        preprocessed_data_path = os.path.join(artifacts_dir, preprocessed_data_dir, preprocessed_data_file)
        target_column_data_path = os.path.join(artifacts_dir, target_column_data_dir, target_column_data_file)
        save_local_df(df, preprocessed_data_path)
        save_local_df(self.target_column, target_column_data_path)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--params", default="config/params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage_03_data_preprocessing started")

        preprocessing_object = preprocessing(config_path=parsed_args.config, params_path=parsed_args.params)
        preprocessing_object.data_preprocessing()

        logging.info("stage_03_data_preprocessing is completed! All the data are saved in local >>>>>")

    except Exception as e:
        logging.exception(e)
        raise e
