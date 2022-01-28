import argparse
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

app = Flask(__name__)
dashboard.bind(app)
CORS(app)
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
    # args = argparse.ArgumentParser()
    # args.add_argument("--params", "-p", default="config/params.yaml")
    # args.add_argument("--config", default="config/config.yaml")
    # args.add_argument("--model", "-m", default="config/model.yaml")
    # parsed_args = args.parse_args()
    config = read_yaml("config/config.yaml")
    params = read_yaml("config/params.yaml")
    database_name = params['logs_database']['database_name']
    training_table_name = params['logs_database']['training_table_name']
    user_name = config['database']['user_name']
    password = config['database']['password']
    db_logs = DBOperations(database_name)
    db_logs.establish_connection(user_name, password)
    logs=([str(tuple(i)).replace('(','').replace(')','') for i in db_logs.show_logs(training_table_name)])
    return render_template("show_training_logs.html", len = len(logs), Logs = logs)
# @app.route("/train", methods=['GET','POST'])
# @cross_origin()
def trainRouteClient():

    try:

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

        #if request.json['folderPath'] is not None:
        get_data(config_path=parsed_args.config, params_path=parsed_args.params)
        preprocessing_object = preprocessing(config_path=parsed_args.config, params_path=parsed_args.params)
        preprocessing_object.data_preprocessing()

        model_training = ModelTraining(config_path=parsed_args.config, params_path=parsed_args.params,
                                           model_path=parsed_args.model)
        model_training.start_model_training()
    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successful!!")
@app.route("/train", methods=['GET','POST'])
@cross_origin()
def scheduler():
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
        # return render_template("index.html")    else:

        t1 = threading.Thread(target=trainRouteClient)
        # t2 = threading.Thread(target=home)
        print(t1.is_alive())
        db_logs.update_model_training_thread_status('R')
        t1.start()
        # t2.start()
        t1.join(3)



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
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
