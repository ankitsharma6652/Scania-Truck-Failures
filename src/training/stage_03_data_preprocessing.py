### @author : Ankit Sharma
from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer,MissingIndicator
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler , Normalizer
from imblearn import over_sampling
from sklearn.decomposition import PCA
import pickle as p
def get_label_column(df,label):
    target_column=df[label]
    df.drop(columns=['class'],inplace=True)
    return target_column


def get_standard_scaling_object():
       # return StandardScaler()
    return Normalizer()
def standard_scaling(df):
    """
    Scaling the data points between the range of 0 and 1
    :param df:  dataframe
    :return:dataframe after standard scaling and standard scaling object
    @author : Ankit Sharma
    """
    std = get_standard_scaling_object()
    train_std=std.fit_transform(df)
    return pd.DataFrame(train_std,columns=df.columns),std
def dimensionality_reduction_using_pca(df,n_components,random_state):
    """
    Performing dimensionality reduction using pca
    selecting 90 features(components) as they are  explaining 97% of data
         @author : Ankit Sharma
    """
    df_pca = PCA(n_components= n_components,random_state=random_state)
    df= df_pca.fit_transform(df)
    return pd.DataFrame(df)
def label_encoding(df):
    """encode labels to 0 and 1"""
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    df = df.copy()
    return df

def remove_highly_corr_featues(df):
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

def upsampling_postive_class(df):
    """
    upsampling the positive class using smote to have balanced dataset.
    @author : Ankit Sharma
    """
    # Upsampling the positive class using Smote Technique
    sm = over_sampling.SMOTE()
    # sm = over_sampling.SMOTE()
    df_Sampled_Smote, y_train = sm.fit_resample(df,df['class'])
    return df_Sampled_Smote
def downsampling_neg_class(df):
    """
    #downsampling the negative class
    @author : Ankit Sharma
    """
    train_neg_sampled = df[df['class'] == 0].sample(n = 10000,random_state = 42)
    train_Sampled = df[df['class'] == 1].append(train_neg_sampled)
    return train_Sampled

def handle_missing_values_using_median_imputation(df):
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
    return df_imputed_median,impute_median
def remove_missing_values_columns(df):
    """
    Replace the na and nan values with np.Nan and Removes the columns where >=70 % values are missing,
    @author : Ankit Sharma
    """
    df.replace(to_replace=['na','nan'],value = np.NaN,inplace=True)
    df.dropna(axis=1, thresh=df.shape[0]*0.7, inplace=True)
    # print(df.columns.shape)
    return df
def data_preprocessing(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    n_components=params['base']['n_components']
    random_state=params['base']['random_state']

    artifacts_dir = config["artifacts"]['artifacts_dir']
    local_data_dirs = config["artifacts"]['local_data_dirs']
    local_data_train_file = config["artifacts"]['local_data_train_file']
    label=params["target_columns"]['columns']
    standard_scaling_file_dir=params['standard_scalar']['standard_scale_file_path']
    standard_scale_file_name=params['standard_scalar']['standard_scale_file_name']
    raw_local_file_path = os.path.join(artifacts_dir, local_data_dirs, local_data_train_file)

    print(raw_local_file_path)

    df = pd.read_csv(raw_local_file_path)
    remove_missing_values_columns(df)
    df=label_encoding(df)
    df=handle_missing_values_using_median_imputation(df)[0]
    # handle_missing_values_using_median_imputation(df)
    df=remove_highly_corr_featues(df)
    df=upsampling_postive_class(downsampling_neg_class(df))
    target_column = get_label_column(df, label)
    df = dimensionality_reduction_using_pca(df, n_components, random_state)
    standard_scalar_data,standard_scaling_object=standard_scaling(df)

    # df=df.merge(target_column)


    # train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    preprocessed_data_dir = config["artifacts"]["preprocessed_data_dir"]
    target_column_data_dir=config['artifacts']['target_column_data_dir']

    create_directory_path([os.path.join(artifacts_dir, preprocessed_data_dir)])
    create_directory_path([os.path.join(artifacts_dir, target_column_data_dir)])
    create_directory_path([os.path.join(artifacts_dir, standard_scaling_file_dir)])

    preprocessed_data_file = config["artifacts"]["preprocessed_data_file"]
    target_column_data_file=config["artifacts"]["target_column_data_file"]
    # target_column_data_dir: target_column_data_dir
    # target_column_data_file: target_column_training_data
    preprocessed_data_path = os.path.join(artifacts_dir, preprocessed_data_dir, preprocessed_data_file)
    target_column_data_path=os.path.join(artifacts_dir, target_column_data_dir, target_column_data_file)
    standard_scaling_data_path=os.path.join(artifacts_dir,standard_scaling_file_dir,standard_scale_file_name)
    save_local_df(standard_scalar_data, preprocessed_data_path)
    save_local_df(target_column, target_column_data_path)
    with open(standard_scaling_data_path,'wb') as s:
        p.dump(standard_scaling_object,s)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="config/params.yaml")

    parsed_args = args.parse_args()

    data_preprocessing(config_path=parsed_args.config, params_path=parsed_args.params)