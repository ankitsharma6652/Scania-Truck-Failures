from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
import argparse
import pandas as pd
import os

def get_data(config_path):

    config = read_yaml(config_path)
    source_download_test_dirs = config["data_source"]["test_data"]
    df_test = pd.read_csv(source_download_test_dirs, sep=",", skiprows=range(0, 20))

    artifacts_dir = config["artifacts"]['artifacts_dir']
    local_data_dirs = config["artifacts"]['local_data_dirs']
    local_data_test_file = config["artifacts"]['local_data_test_file']
    local_data_dir_path = os.path.join(artifacts_dir, local_data_dirs)

    create_directory_path(dirs=[local_data_dir_path])
    local_data_test_file_path = os.path.join(local_data_dir_path, local_data_test_file)
    df_test.to_csv(local_data_test_file_path, sep=",", index=False)

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    get_data(config_path=parsed_args.config)
