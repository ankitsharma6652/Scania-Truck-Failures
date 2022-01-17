from src.utils.all_utils import read_yaml, create_directory_path, read_yaml
import argparse
import os
import logging
import pandas as pd
# import s3fs

logging_str = "[%(asctime)s: %(levelname)s %(module)s]: %(message)s"
log_dir = "../../logs"

os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str, filemode='a')


def get_data(config_path):
    config = read_yaml(config_path)

    source_download_train_dirs = config["data_source"]["s3_source_train"]
    source_download_test_dirs = config["data_source"]["s3_source_test"]
    df_train = pd.read_csv(source_download_train_dirs, sep=",",skiprows=range(0,20))
    df_test = pd.read_csv(source_download_test_dirs, sep=",",skiprows=range(0,20))

    artifacts_dir = config["artifacts"]['artifacts_dir']
    local_data_dirs = config["artifacts"]['local_data_dirs']
    local_data_train_file = config["artifacts"]['local_data_train_file']
    local_data_test_file = config["artifacts"]['local_data_test_file']

    local_data_dir_path = os.path.join(artifacts_dir, local_data_dirs)

    create_directory_path(dirs= [local_data_dir_path ])

    local_data_train_file_path = os.path.join(local_data_dir_path , local_data_train_file)
    local_data_test_file_path = os.path.join(local_data_dir_path , local_data_test_file)
    
    df_train.to_csv(local_data_train_file_path, sep=",", index=False)
    df_test.to_csv(local_data_test_file_path, sep=",", index=False)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage_00_data_loader started")
        get_data(config_path=parsed_args.config)
        logging.info("stage_00_data_loader completed! All the data are saved in local >>>>>")
    except Exception as e:
        logging.exception(e)
        raise e