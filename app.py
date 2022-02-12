import argparse
import smtplib
from email.message import EmailMessage
from src.utils.email_sender.email_sender import email_sender
from src.utils.all_utils import read_yaml
from src.training.stage_01_data_loader import get_data
from src.training.stage_02_data_preprocessing import preprocessing
from src.training.stage_03_model_training import ModelTraining
from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
from src.utils.DbOperations_Logs import DBOperations
import threading
import schedule
import pandas as pd
import json
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
from flask import Flask, jsonify, request
import json, os, signal
app = Flask(__name__)
dashboard.bind(app)
CORS(app)

def stopServer():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({ "success": True, "message": "Server is shutting down..." })
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')
@app.route("/scheduler_manager", methods=['GET'])
@cross_origin()
def scheduler_manager():
    return render_template('scheduler_manager.html')
@app.route("/show_training_logs", methods=['GET','POST'])
@cross_origin()
def show_training_logs():
    config = read_yaml("config/config.yaml")
    params = read_yaml("config/params.yaml")
    database_name = params['logs_database']['database_name']
    training_table_name = params['logs_database']['training_table_name']
    user_name = config['database']['user_name']
    password = config['database']['password']
    db_logs = DBOperations(database_name)
    db_logs.establish_connection(user_name, password)
    stage_name = []
    time = []
    method_name = []
    Logs = []
    for i in db_logs.show_logs(training_table_name):
        stage_name.append(i[0])
        time.append(i[1])
        method_name.append(i[2])
        Logs.append(i[3])
    return render_template("show_training_logs.html", len = len(Logs), stage_name=stage_name,time=time,method_name=method_name,Logs = Logs)
# @app.route("/train", methods=['GET','POST'])
# @cross_origin()
def trainRouteClient(recievers_email):

    try:
        print("Train Route Client",type(recievers_email))
        config = read_yaml("config/config.yaml")
        params = read_yaml("config/params.yaml")

        args = argparse.ArgumentParser()
        args.add_argument("--params", "-p", default="config/params.yaml")
        args.add_argument("--config", default="config/config.yaml")
        args.add_argument("--model", "-m", default="config/model.yaml")
        parsed_args = args.parse_args()
        database_name = params['logs_database']['database_name']
        training_table_name = params['logs_database']['training_table_name']
        model_training_thread_table_name=params['model_training_thread']['model_training_thread_table_name']

        user_name = config['database']['user_name']
        password = config['database']['password']
        db_logs = DBOperations(database_name)
        db_logs.establish_connection(user_name, password)
        db_logs.model_training_thread(model_training_thread_table_name)
        print('Updated Status to Running')
        # print(request.json)
        #if request.json['folderPath'] is not None:
        db_logs.update_model_training_thread_status('R')
        # get_data(config_path=parsed_args.config, params_path=parsed_args.params)
        # preprocessing_object = preprocessing(config_path=parsed_args.config, params_path=parsed_args.params)
        # preprocessing_object.data_preprocessing()
        # print("Email",request.form['email'])
        model_training = ModelTraining(config_path=parsed_args.config, params_path=parsed_args.params,
                                           model_path=parsed_args.model,recievers_email=recievers_email)
        model_training.start_model_training()

        stopServer()

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successful!!")
@app.route("/train", methods=['GET','POST'])
@cross_origin()
def start_training():

    if len(request.form["email"]) == 0:
        return render_template("model_training.html")
    else:
        es=email_sender()
        email_address=request.form["email"]
        es.set_reciever_mail(email_address)
        print("Reciever's mail",es.get_reciever_mail())
        scheduler(email_address)
        return render_template("send_email.html",email_address=email_address)
@app.route("/start_training", methods=['GET','POST'])
@cross_origin()
def training():
    config = read_yaml("config/config.yaml")
    params = read_yaml("config/params.yaml")

    args = argparse.ArgumentParser()
    args.add_argument("--params", "-p", default="config/params.yaml")
    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--model", "-m", default="config/model.yaml")
    parsed_args = args.parse_args()
    database_name = params['logs_database']['database_name']
    training_table_name = params['logs_database']['training_table_name']
    model_training_thread_table_name = params['model_training_thread']['model_training_thread_table_name']

    user_name = config['database']['user_name']
    password = config['database']['password']
    db_logs = DBOperations(database_name)
    db_logs.establish_connection(user_name, password)
    db_logs.model_training_thread(model_training_thread_table_name)
    if ('R') in list(db_logs.model_training_thread_status()):
        return Response("Model Training in Progress, Please try later")
    else:

        return render_template("model_training.html")
# @app.route("/train", methods=['GET','POST'])
# @cross_origin()
def scheduler(email_address):
    config = read_yaml("config/config.yaml")
    params = read_yaml("config/params.yaml")

    args = argparse.ArgumentParser()
    args.add_argument("--params", "-p", default="config/params.yaml")
    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--model", "-m", default="config/model.yaml")
    parsed_args = args.parse_args()
    database_name = params['logs_database']['database_name']
    training_table_name = params['logs_database']['training_table_name']
    model_training_thread_table_name = params['model_training_thread']['model_training_thread_table_name']

    user_name = config['database']['user_name']
    password = config['database']['password']
    db_logs = DBOperations(database_name)
    db_logs.establish_connection(user_name, password)
    db_logs.model_training_thread(model_training_thread_table_name)

    # if request.json['folderPath'] is not None:
    # if 1 in db_logs.model_training_thread_status():
    #     return Response("Model Training in Progress, Please try later")
    # else:

    print(db_logs.model_training_thread_status())
    if ('R') in list(db_logs.model_training_thread_status()):
        return Response("Model Training in Progress, Please try later")
        # return render_template("index.html")
    else:
        print(email_address)
        t1 = threading.Thread(target=trainRouteClient,args=[email_address])
        # t2 = threading.Thread(target=home)
        print(t1.is_alive())
        # db_logs.update_model_training_thread_status('R')
        t1.start()
        # t2.start()
        t1.join(3)
        print(request.json)


        # t2.join(3)


        return render_template("model_training.html")
# @app.route("/schedule_manager", methods=['GET','POST'])
# @cross_origin()
# def schedule_manager():
#     return render_template("scheduler_manager.html")
port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # port = 5000
    # print(email_address)
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()

