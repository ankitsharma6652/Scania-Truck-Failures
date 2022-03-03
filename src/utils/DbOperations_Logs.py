from src.utils.all_utils import read_yaml, create_directory_path, save_local_df
import argparse
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import datetime
from src.utils import logger

def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

def unix_time_millis(dt):
    return int(unix_time(dt))*1000
class DBOperations:
    """
    This class shall be used to perform the Database operation for the Logs and the insert_logs() method will be used to implementing the local logging.
    @author: Ankit Sharma

    """
    def __init__(self,database_name):
        self.database_name=database_name
        
    def get_counter(self):
        global counter
        counter=counter+1
        yield counter

    def establish_connection(self,username,password):
        try :
            self.cloud_config = {
                'secure_connect_bundle': r'D:\Data_Science\Scania-Truck-Failures\artifacts\secure-connect-scania-truck-failures.zip'
            }
            self.auth_provider = PlainTextAuthProvider(username,password)
            self.cluster = Cluster(cloud=self.cloud_config, auth_provider=self.auth_provider,protocol_version=4)
            self.session = self.cluster.connect()

            row = self.session.execute("select release_version from system.local").one()
            if row:
                print(row[0])
            else:
                print("An error occurred.")

            return "Connection established"
        except Exception as e:
            print(e)

            raise Exception("Error in connection establishment with Database",e)
            return e
    

    # def establish_connection(self,config_path):    
    #     try:
    #         config = read_yaml(config_path)
    #         self.cloud_config = {
    #         'secure_connect_bundle': config['database']['path']
    #     }
    #         self.auth_provider = PlainTextAuthProvider(config['database']['user_name'], config['database']['password'])

    #         self.cluster = Cluster(cloud=self.cloud_config, auth_provider=self.auth_provider,protocol_version=4)
    #         self.session = self.cluster.connect()

    #         row = self.session.execute("select release_version from system.local").one()
    #         if row:
    #             print(row[0])
    #         else:
    #             print("An error occurred.")

    #         return "Connection established"
    #     except Exception as e:
    #         print(e)
    #         raise Exception("Error in connection establishment with Database",e)
        
    def create_table(self,table_name):
        try:
            return self.session.execute(f"create table if not exists {self.database_name}.{table_name}(time timestamp, stage_name text, method_name text,logs text ,primary key(stage_name,time,method_name,logs)) WITH CLUSTERING ORDER BY (time desc,method_name asc,logs asc)")
            # print("Executed Success")
        except Exception as e:
            # print(e)
            return e

    def insert_logs(self,table_name,stage_name,method_name,logs):
        try:
            # global counter
            # counter=counter+1
            query=f"insert into {self.database_name}.{table_name}(time,stage_name,method_name,logs) values({unix_time_millis(datetime.datetime.now())},'{stage_name}','{method_name}','{logs}')"
            logger.App_Logger().log(table_name,logs)
            return  self.session.execute(query)
        except Exception as e:
            print(e)
            return e

    def show_logs(self,table_name):
        try:
            return self.session.execute(f"select * from {self.database_name}.{table_name} ")
        except Exception as e:
            print(e)
            return e

    def model_training_thread(self,tablename):
      
        try:
            result=self.session.execute(f"select table_name from  system_schema.tables WHERE keyspace_name = '{self.database_name}' and table_name='{tablename}'")
            if tablename in list(result.one()):
                return "Table exists"
            else:
                self.session.execute(
                    f"create table if not exists {self.database_name}.{tablename}(id int primary key,status text )")
                print("Executed Success")
                return self.session.execute(f"insert into {self.database_name}.{tablename}(id,status) values (1,'NS')")
        except Exception as e:
            print(e)
            raise Exception(e)
            return e

    def update_model_training_thread_status(self,status):
        self.session.execute(f"update {self.database_name}.model_training_thread set status = '{status}' where id =1")
        return f"Status update to {status}"

    def model_training_thread_status(self):
        return self.session.execute(f"select status from {self.database_name}.model_training_thread").one()

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="config/params.yaml")
    parsed_args = args.parse_args()
    database_name='scania_truck_failures'
    DB=DBOperations(database_name)
    DB.establish_connection(config_path=parsed_args.config)
    DB.create_table('scania_training')



