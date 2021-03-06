from src.utils.all_utils import read_yaml, create_directory_path, read_yaml
import argparse
import os
import pandas as pd
from src.utils.DbOperations_Logs import DBOperations

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

    except Exception as e:
        print(e)
        db_logs.insert_logs(prediction_table_name, stage_name, "get_data", e)
        return e
        
if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--params", default="config/params.yaml")
    args.add_argument("--config", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        get_data(config_path=parsed_args.config, params_path=parsed_args.params)
    except Exception as e:
        raise e
