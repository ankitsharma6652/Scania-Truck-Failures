from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
import argparse
import pandas as pd
import os
from src.utils.DbOperations_Logs import DBOperations

def get_data(config_path, params_path):
    stage_name = os.path.basename(__file__)[:-3]
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    database_name = params['logs_database']['database_name']
    prediction_table_name = params['logs_database']['prediction_table_name']
    user_name = config['database']['user_name']
    password = config['database']['password']
    db_logs = DBOperations(database_name)
    db_logs.establish_connection(user_name,password)
    db_logs.create_table(prediction_table_name)

    try:
        source_test_dirs = config["data_source"]["source_test"]
        df_test = pd.read_csv(source_test_dirs, sep=",", skiprows=range(0, 20))

        db_logs.insert_logs(prediction_table_name,stage_name,"get_data","Testing data downloading start")
        df_test = pd.read_csv(source_test_dirs, sep=",",skiprows=range(0,20))
        db_logs.insert_logs(prediction_table_name, stage_name, "get_data", "Testing data downloading completed")
        
        artifacts_dir = config["artifacts"]['artifacts_dir']
        local_data_dirs = config["artifacts"]['local_data_dirs']
        local_data_test_file = config["artifacts"]['local_data_test_file']
        local_data_dir_path = os.path.join(artifacts_dir, local_data_dirs)

        create_directory_path(dirs=[local_data_dir_path])

        local_data_test_file_path = os.path.join(local_data_dir_path, local_data_test_file)
        db_logs.insert_logs(prediction_table_name, stage_name, "get_data", f"Testing data file saved at the Location: {local_data_test_file_path}")
        df_test.to_csv(local_data_test_file_path, sep=",", index=False)

    except Exception as e:
        print(e)
        db_logs.insert_logs(prediction_table_name, stage_name, "get_data", f"{e}")
        raise Exception

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--params", default="config/params.yaml")
    args.add_argument("--config", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        get_data(config_path=parsed_args.config, params_path=parsed_args.params)
        
    except Exception as e:
        raise e    


