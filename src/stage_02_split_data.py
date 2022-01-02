from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer,MissingIndicator
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn import over_sampling

def label_encoding(df):
    #encode labels to 0 and 1
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    df = df.copy()
    return df

    
def upsampling_postive_class(df):
    """
    upsampling the positive class using smote to have balanced dataset.
    :param df:
    :return:
    """
    # Upsampling the positive class using Smote Technique
    sm = over_sampling.SMOTE()
    # sm = over_sampling.SMOTE()
    df_Sampled_Smote, y_train = sm.fit_resample(df,df['class'])
    return df_Sampled_Smote
def downsampling_neg_class(df):
    """
    #downsampling the negative class
    :param df:
    :return:
    """
    train_neg_sampled = df[df['class'] == 0].sample(n = 10000,random_state = 42)
    train_Sampled = df[df['class'] == 1].append(train_neg_sampled)
    return train_Sampled

def handle_missing_values_using_median_imputation(df):
    """
    fill the missing values by using Median Imputation .
    :return:imputed dataframe, imputed object
    """

    # df1=df.drop(columns=['class'])
    impute_median = SimpleImputer(missing_values=np.nan, strategy='median', copy=True, verbose=2)
    df_imputed_median = pd.DataFrame(impute_median.fit_transform(df), columns=df.columns)
    # print(df_imputed_median.isna().sum().sum())
    # print(df_imputed_median.shape)
    # df_imputed_median['class']=df['class']
    return df_imputed_median,impute_median
def remove_missing_values_columns(df):
    """
    Repalce the na and nan values with np.Nan and Removes the columns where >=70 % values are missing,
    @author : Ankit Sharma
    """
    df.replace(to_replace=['na','nan'],value = np.NaN,inplace=True)
    df.dropna(axis=1, thresh=df.shape[0]*0.7, inplace=True)
    # print(df.columns.shape)
    return df
def split_and_save(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts_dir = config["artifacts"]['artifacts_dir']
    local_data_dirs = config["artifacts"]['local_data_dirs']
    local_data_train_file = config["artifacts"]['local_data_train_file']

    raw_local_file_path = os.path.join(artifacts_dir, local_data_dirs, local_data_train_file)

    print(raw_local_file_path)
    
    df = pd.read_csv(raw_local_file_path)

    split_ratio = params["base"]["test_size"]
    random_state = params["base"]["random_state"]
    remove_missing_values_columns(df)
    df=label_encoding(df)
    df=handle_missing_values_using_median_imputation(df)[0]
    # handle_missing_values_using_median_imputation(df)
    df=upsampling_postive_class(downsampling_neg_class(df))
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    split_data_dir = config["artifacts"]["split_data_dir"]

    create_directory_path([os.path.join(artifacts_dir, split_data_dir)])

    train_data_filename = config["artifacts"]["train"]
    test_data_filename = config["artifacts"]["test"]


    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename)
    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename)

    for data, data_path in (train, train_data_path), (test, test_data_path):
        save_local_df(data, data_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="config/params.yaml")

    parsed_args = args.parse_args()

    split_and_save(config_path=parsed_args.config, params_path=parsed_args.params)