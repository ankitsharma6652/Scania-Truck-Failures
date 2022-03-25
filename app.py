import argparse
import smtplib
from email.message import EmailMessage
from src.utils.email_sender.email_sender import email_sender
from src.utils.all_utils import read_yaml
from src.utils.KThread import KThread
from src.training.stage_01_data_loader import get_data
from src.training.stage_02_data_preprocessing import preprocessing
from src.training.stage_03_model_training import ModelTraining
# from src.prediction.stage_01_data_loader import get_data as prediction_get_data
from src.prediction.stage_02_data_preprocessing import preprocessing as prediction_preprocessing
from src.prediction.stage_03_model_predictor import Predictor
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

application = Flask(__name__)
dashboard.bind(application)
CORS(application)
isAlive=False
global t1
t1=""


def stopServer():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({ "success": True, "message": "Server is shutting down..." })
@application.route("/download_file", methods=['GET'])
@cross_origin()
def download_file():
    config = read_yaml("config/config.yaml")
    params = read_yaml("config/params.yaml")

    args = argparse.ArgumentParser()
    args.add_argument("--params", "-p", default="config/params.yaml")
    args.add_argument("--config", default="config/config.yaml")
    args.add_argument("--model", "-m", default="config/model.yaml")
    parsed_args = args.parse_args()
    predictor = Predictor(config_path=parsed_args.config, params_path=parsed_args.params,
                          model_path=parsed_args.model)
    downloaded=predictor.download_prediction_file()
    return Response(f"File is downloaded:{downloaded}")
@application.route("/prediction_page", methods=['GET'])
@cross_origin()
def prediction_page():
    return render_template('prediction_page.html')
@application.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('homepage.html')


@application.route("/scheduler_manager", methods=['GET'])
@cross_origin()
def scheduler_manager():
    return render_template('scheduler_manager.html')
@application.route("/show_training_logs", methods=['GET','POST'])
@cross_origin()
def show_training_logs():
    '''This function shall be used to Show Training Logs'''
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
@application.route("/show_prediction_logs", methods=['GET','POST'])
@cross_origin()
def prediction_logs():
    '''
    This function shall be used to show prediction logs
    :return:
    '''
    config = read_yaml("config/config.yaml")
    params = read_yaml("config/params.yaml")
    database_name = params['logs_database']['database_name']
    # training_table_name = params['logs_database']['training_table_name']
    prediction_table_name = params['logs_database']['prediction_table_name']
    user_name = config['database']['user_name']
    password = config['database']['password']
    db_logs = DBOperations(database_name)
    db_logs.establish_connection(user_name, password)
    stage_name = []
    time = []
    method_name = []
    Logs = []
    for i in db_logs.show_logs(prediction_table_name):
        stage_name.append(i[0])
        time.append(i[1])
        method_name.append(i[2])
        Logs.append(i[3])
    return render_template("show_prediction_logs.html", len = len(Logs), stage_name=stage_name,time=time,method_name=method_name,Logs = Logs)
@application.route("/predict", methods=['GET'])
@cross_origin()
def prediction():

    try:
        # print("Train Route Client",type(recievers_email))
        config = read_yaml("config/config.yaml")
        params = read_yaml("config/params.yaml")

        args = argparse.ArgumentParser()
        args.add_argument("--params", "-p", default="config/params.yaml")
        args.add_argument("--config", default="config/config.yaml")
        args.add_argument("--model", "-m", default="config/model.yaml")
        parsed_args = args.parse_args()
        # database_name = params['logs_database']['database_name']
        # training_table_name = params['logs_database']['training_table_name']
        # model_training_thread_table_name=params['model_training_thread']['model_training_thread_table_name']
        #
        # user_name = config['database']['user_name']
        # password = config['database']['password']
        # db_logs = DBOperations(database_name)
        # db_logs.establish_connection(user_name, password)
        # db_logs.model_training_thread(model_training_thread_table_name)
        # print('Updated Status to Running')
        # print(request.json)
        #if request.json['folderPath'] is not None:
        # db_logs.update_model_training_thread_status('R')
        # get_data(config_path=parsed_args.config, params_path=parsed_args.params)
        # preprocessing_object = preprocessing(config_path=parsed_args.config, params_path=parsed_args.params)
        # preprocessing_object.data_preprocessing()
        # print("Email",request.form['email'])
        # prediction_get_data(config_path=parsed_args.config, params_path=parsed_args.params)
        # pred_preprocessing=prediction_preprocessing(config_path=parsed_args.config, params_path=parsed_args.params)
        # pred_preprocessing.data_preprocessing()
        predictor=Predictor(config_path=parsed_args.config, params_path=parsed_args.params,
                                           model_path=parsed_args.model)
        prediction_file_location,df=predictor.predict()
        return render_template('prediction.html',prediction_file_location=prediction_file_location,sample_output=df,len=(df.shape[0]),df_columns=df.columns)



    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    # return Response("Prediction successful!!")
@application.route("/start_training_again", methods=['GET','POST'])
@cross_origin()
def training_status():
    """
    This function shall be used to show training status on UI and also to kill the running thread if user demands to start the training again
    :return:
    """
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
    # print('Updated Status to NS')
    # # # print(request.json)
    # # # if request.json['folderPath'] is not None:
    # db_logs.update_model_training_thread_status('NS')
    if isAlive:
        if t1.is_alive():
            t1.kill()
            print('Updated Status to NS in is_alive block')
            globals()['isAlive'] = False
            db_logs.update_model_training_thread_status('NS')
            return render_template("model_training.html")
    else:
        print('Updated Status to NS in else block')
        db_logs.update_model_training_thread_status('NS')
        return render_template("model_training.html")
    return render_template("model_training.html")
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
        db_logs.update_model_training_thread_status('R')

        print('Updated Status to Running')
        # print(request.json)
        #if request.json['folderPath'] is not None:
        db_logs.update_model_training_thread_status('R')
        print('Hello world')
        # get_data(config_path=parsed_args.config, params_path=parsed_args.params)
        # preprocessing_object = preprocessing(config_path=parsed_args.config, params_path=parsed_args.params)
        # preprocessing_object.data_preprocessing()
        # print("Email",request.form['email'])
        model_training = ModelTraining(config_path=parsed_args.config, params_path=parsed_args.params,
                                           model_path=parsed_args.model)
        mail_text=model_training.start_model_training()
        email_sender().send_email(mail_text=mail_text, TO=recievers_email)
        print("email sent",recievers_email)

        # stopServer()

    except ValueError as e:
        print(e)
        return Response("Error Occurred! %s" % ValueError)

    except KeyError as e:
        print(e)
        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:
        print(e)
        return Response("Error Occurred! %s" % e)
    return Response("Training successful!!")
@application.route("/train", methods=['GET','POST'])
@cross_origin()
def start_training():

    if len(request.form["email"]) == 0:
        return render_template("model_training.html")
    else:
        es=email_sender()
        email_address=request.form["email"]
        es.set_reciever_mail(email_address)
        print("Reciever's mail",es.get_reciever_mail())
        if es.validate_email(email_address):
            es.notify_email(email_address)
            scheduler(email_address)
            return render_template("send_email.html",email_address=email_address)
        else:
            return render_template("email_address_validator.html")
@application.route("/start_training", methods=['GET','POST'])
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
    # db_logs.update_model_training_thread_status('NS')
    # print("Thread is alive",is_alive)
    db_logs.model_training_thread(model_training_thread_table_name)
    if ('R') in list(db_logs.model_training_thread_status()):
        return render_template("training_status.html")
    else:

        return render_template("model_training.html")
# @app.route("/train", methods=['GET','POST'])
# @cross_origin()
@application.route("/training_page", methods=['GET'])
@cross_origin()
def training_page():
    return render_template('training.html')


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
        # return render_template("training.html")
    else:
        print(email_address)
        # t1 = threading.Thread(target=trainRouteClient,args=[email_address])
        # # t2 = threading.Thread(target=home)
        # print(t1.is_alive())
        #
        # # db_logs.update_model_training_thread_status('R')
        # t1.start()
        # print("Thread_status",t1.is_alive())
        # globals()['is_alive']=True
        # # t2.start()
        # t1.join(3)
        # print(request.json)
        globals()['t1'] = KThread(target=trainRouteClient, args=[email_address])
        # t2 = threading.Thread(target=home)
        print(t1.is_alive())

        # db_logs.update_model_training_thread_status('R')
        t1.start()
        print("Thread_status", t1.is_alive())
        globals()['isAlive'] = t1.is_alive()
        # t2.start()
        t1.join(3)
        print(request.json)


        # t2.join(3)


        return render_template("model_training.html")
port = int(os.getenv("PORT",8000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # # port = 5000
    # # print(email_address)
    httpd = simple_server.make_server(host, port, application)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
    # application.run(debug=True, use_reloader=True)
# web: gunicorn app:application -b 0.0.0.0:8000 -w 3
# web: python app.py   --master --processes 4 --threads 2