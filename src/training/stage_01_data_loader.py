from src.utils.all_utils import read_yaml, create_directory_path, read_yaml
import argparse
import os,sys
import pandas as pd
from src.utils.DbOperations_Logs import DBOperations
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from src.exception import CustomException

def get_data(config_path,params_path):
    # print("Inside Get Data Function")
    stage_name = os.path.basename(__file__)[:-3]
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    database_name = params['logs_database']['database_name']
    training_table_name = params['logs_database']['training_table_name']
    user_name=config['database']['user_name']
    password=config['database']['password']
    db_logs = DBOperations(database_name)

    # aws = AmazonSimpleStorageService('AKIAXZVACUKMP6ZTKK3E', 'veUIBpOnwohAe5Wc5pGzCMtJfT0u+fsFZFlQsFAc', 'scania-121')

    db_logs.establish_connection(user_name,password)

    db_logs.create_table(training_table_name)
    access_key, secret_access_key = db_logs.get_aws_s3_keys()
    print(access_key,secret_access_key)
    aws=AmazonSimpleStorageService(access_key,secret_access_key,config['storage']['bucket_name'])

    try:

        source_download_train_dirs = config["data_source"]["train_data"]
        # print(database_name)
        # db_logs.insert_logs(training_table_name,stage_name,"get_data","Training data downloading start")
        # df_train = pd.read_csv(source_download_train_dirs, sep=",",skiprows=range(0,20))
        # db_logs.insert_logs(training_table_name, stage_name, "get_data", "Training data downloading completed")

        artifacts_dir = config["artifacts"]['artifacts_dir']
        local_data_dirs = config["artifacts"]['local_data_dirs']
        local_data_train_file = config["artifacts"]['local_data_train_file']

        local_data_dir_path = os.path.join(artifacts_dir, local_data_dirs)

        create_directory_path(dirs= [local_data_dir_path ])

        local_data_train_file_path = os.path.join(local_data_dir_path , local_data_train_file)
        if not os.path.exists(local_data_train_file_path):
            db_logs.insert_logs(training_table_name, stage_name, "get_data", "Training data downloading start")
            df_train = pd.read_csv(source_download_train_dirs, sep=",", skiprows=range(0, 20))
            db_logs.insert_logs(training_table_name, stage_name, "get_data", "Training data downloading completed")

            df_train.to_csv(local_data_train_file_path, sep=",", index=False)
            db_logs.insert_logs(training_table_name, stage_name, "get_data", f"Training data file saved at the Location: {local_data_train_file_path}")
        else:
            print("Training Data Already exists")
            # aws.write_file_content(r'artifacts/local_data_dirs',local_data_train_file_path,df_train)
            db_logs.insert_logs(training_table_name, stage_name, "get_data", f"Training data Already exists at location {local_data_train_file_path}")
        # print(local_data_dir_path,local_data_train_file)
        # print(aws.is_file_present(local_data_dir_path.replace("\\","/"),local_data_train_file))
        if not  aws.is_file_present(local_data_dir_path.replace("\\","/"),local_data_train_file)['status']:
            db_logs.insert_logs(training_table_name, stage_name, "get_data", f"Training dataset uploaded to s3 storage at location {local_data_train_file_path}")

            aws.upload_file(local_data_dir_path.replace("\\","/"),local_data_train_file,local_data_train_file,local_file_path=local_data_train_file_path)
        else:
            db_logs.insert_logs(training_table_name, stage_name, "get_data", f"Training dataset Already exists in s3 storage at location {local_data_train_file_path}")

    except Exception as e:
        print(e)
        db_logs.insert_logs(training_table_name, stage_name, "get_data", f"{e}")
        raise (CustomException(e, sys)) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--params", default="config/params.yaml")
    args.add_argument("--config", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        get_data(config_path=parsed_args.config, params_path=parsed_args.params)
    except Exception as e:
        raise e