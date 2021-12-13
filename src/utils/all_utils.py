import yaml
import os
import json


def read_params(config_path: str) -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config



def create_directory_path(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"directory is created at {dir_path}")


def save_local_df(data, data_path, index_status=False):
    data.to_csv(data_path, index=index_status)
    print(f"data is saved at {data_path}")


def save_reports(report: dict, report_path: str, indentation=4):
    with open(report_path, "w") as f:
        json.dump(report, f, indent=indentation)
    print(f"reports are saved at {report_path}")