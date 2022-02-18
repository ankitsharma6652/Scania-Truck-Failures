import yaml
import os
import json
import logging
import time
import pandas


from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
from email.message import EmailMessage


import smtplib

def read_yaml(config_path: str) -> dict:
    with open(config_path) as yaml_file:
        content = yaml.safe_load(yaml_file)
        logging.info(f"yaml file:{config_path} loaded successfully")
    return content

def create_directory_path(dirs: list) -> None:
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"directory is created at {dir_path}")


def save_local_df(data, data_path, index_status=False):
    data.to_csv(data_path, index=index_status)
    logging.info(f"data is saved at {data_path}")


def save_reports(report: dict, report_path: str, indentation=4):
    with open(report_path, "w") as f:
        json.dump(report, f, indent=indentation)
    logging.info(f"reports are saved at {report_path}")


def send_email(to,mail_text):
        """
        message: Message string in html format
        subject: subject of email
        """
        try:
            print(request.json)
            message = EmailMessage()
            message["Subject"] = "Scania Truck Failures Training Completed"
            message["From"] = "ankitcoolji@gmail.com"
            message["To"] = to
            text = f"Hi recipient,\n\n This is notification email from Scania Truck Failures Application.\n\n" \
                   f"Description: \n\n{mail_text} \n\nThanks & Regards," \
                   f"\nAnkit Sharma"
            message.set_content(text)
            # Create secure connection with server and send email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login("ankitcoolji@gmail.com", "ltcftqdsqssptlbu")
                smtp.send_message(message)
            return Response(request.json)
        except Exception as e:
            print(e)
            raise Exception(e)
