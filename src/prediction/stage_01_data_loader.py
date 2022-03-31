from src.utils.all_utils import read_yaml, create_directory_path, read_yaml
import argparse
import os,sys
import pandas as pd
from src.utils.DbOperations_Logs import DBOperations
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from src.exception import CustomException
def get_data(config_path,params_path):
    stage_name = os.path.basename(__file__)[:-3]
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    database_name = params['logs_database']['database_name']
    prediction_table_name = params['logs_database']['prediction_table_name']
    user_name = config['database']['user_name']
    password = config['database']['password']
    db_logs = DBOperations(database_name)

    db_logs.establish_connection(user_name, password)
    db_logs.create_table(prediction_table_name)
    config = read_yaml(config_path)
    access_key, secret_access_key = db_logs.get_aws_s3_keys()
    print(access_key, secret_access_key)
    aws = AmazonSimpleStorageService(access_key, secret_access_key, config['storage']['bucket_name'])

    try:

        source_download_test_dirs = config["data_source"]["test_data"]
        # db_logs.insert_logs(prediction_table_name, stage_name, "get_data", "Test data downloading start")
        # df_test = pd.read_csv(source_download_test_dirs, sep=",", skiprows=range(0, 20))
        # db_logs.insert_logs(prediction_table_name, stage_name, "get_data", "Test data downloading completed")

        artifacts_dir = config["artifacts"]['artifacts_dir']
        local_data_dirs = config["artifacts"]['local_data_dirs']
        local_data_test_file = config["artifacts"]['local_data_test_file']
        local_data_dir_path = os.path.join(artifacts_dir, local_data_dirs)

        create_directory_path(dirs=[local_data_dir_path])
        local_data_test_file_path = os.path.join(local_data_dir_path, local_data_test_file)
        if not os.path.exists(local_data_test_file_path):
            db_logs.insert_logs(prediction_table_name, stage_name, "get_data", "Test data downloading start")
            df_test = pd.read_csv(source_download_test_dirs, sep=",", skiprows=range(0, 20))
            db_logs.insert_logs(prediction_table_name, stage_name, "get_data", "Test data downloading completed")
            df_test.to_csv(local_data_test_file_path, sep=",", index=False)
            db_logs.insert_logs(prediction_table_name, stage_name, "get_data", f"Test file saved at:{local_data_test_file_path}")
        else:
            print("Test Data Exists")
            db_logs.insert_logs(prediction_table_name, stage_name, "get_data",
                                f"Test file already exists  at location:{local_data_test_file_path}")
        if not  aws.is_file_present(os.path.join(artifacts_dir, local_data_dirs).replace("\\","/"),local_data_test_file)['status']:
            db_logs.insert_logs(prediction_table_name, stage_name, "get_data", f"Training dataset uploaded to s3 storage at location {local_data_test_file_path}")

            aws.upload_file(os.path.join(artifacts_dir, local_data_dirs).replace("\\","/"),local_data_test_file,local_data_test_file,local_file_path=local_data_test_file_path)
        else:
            db_logs.insert_logs(prediction_table_name, stage_name, "get_data", f"Test dataset Already exists in s3 storage at location {local_data_test_file_path}")
    except Exception as e:
        print(e)
        db_logs.insert_logs(prediction_table_name, stage_name, "get_data", e)
        # return e
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
