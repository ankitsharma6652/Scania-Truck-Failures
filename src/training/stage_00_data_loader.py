from src.utils.all_utils import read_params, create_directory_path
import argparse
import os
import shutil
from tqdm import tqdm
import logging
import pandas as pd

logging_str = "[%(asctime)s: %(levelname)s %(module)s]: %(message)s"
log_dir = "logs"

os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str, filemode='a')

# def copy_file(source_download_dir, local_data_dir):
#     list_of_files = os.listdir(source_download_dir)
#     N = len(list_of_files)

#     for file in tqdm(list_of_files, total=N, desc=f'copying file from {source_download_dir} to {local_data_dir}', colour="green" ):
#         src = os.path.join(source_download_dir, file)
#         dest = os.path.join(local_data_dir, file)
#         shutil.copy(src, dest)
        

def get_data(config_path):
    config = read_params(config_path)

    source_download_dirs = config["source_download_dirs"]
    df = pd.read_csv(source_download_dirs, sep=",")

    artifacts_dir = config["artifacts"]['artifacts_dir']
    local_data_dirs = config["artifacts"]['local_data_dirs']
    local_data_file = config["artifacts"]['local_data_file']

    local_data_dir_path = os.path.join(artifacts_dir, local_data_dirs)

    create_directory_path(dirs= [local_data_dir_path ])

    local_data_file_path = os.path.join(local_data_dir_path , local_data_file)
    
    df.to_csv(local_data_file_path, sep=",", index=False)

    
    # for source_download_dir, local_data_dir in tqdm(zip(source_download_dirs, local_data_dirs), total=2, desc= "list of folders", colour="red"):
    #     create_directory_path([local_data_dirs])
    #     copy_file(source_download_dirs, local_data_dirs)

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