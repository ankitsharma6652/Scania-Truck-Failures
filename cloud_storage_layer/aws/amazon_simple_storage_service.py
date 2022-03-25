"""
AWS SDK for Python (Boto3) to create, configure, and manage AWS services,
such as Amazon Elastic Compute Cloud (Amazon EC2) and Amazon Simple Storage Service (Amazon S3)
"""
import json
import boto3
from cloud_storage_layer.aws_exception import AWSException
import sys, os
import dill
import io
import pandas as pd
import pickle
from io import BytesIO
class AmazonSimpleStorageService:

    def __init__(self, access_key_id,secret_access_key, bucket_name,region_name=None):
        """

        :param bucket_name:specify bucket name, if s3 bucket does not exists, new bucket will be created
        :param region_name: specify region name
        """
        try:


            self.bucket_name = bucket_name
            if region_name is None:
                self.client = boto3.client('s3',
                                           aws_access_key_id=access_key_id,
                                           aws_secret_access_key=secret_access_key,
                                           )
                self.resource = boto3.resource('s3',
                                               aws_access_key_id=access_key_id,
                                               aws_secret_access_key=secret_access_key,
                                               region_name=region_name
                                               )
            else:
                self.client = boto3.client('s3',
                                           aws_access_key_id=access_key_id,
                                           aws_secret_access_key=secret_access_key,
                                           region_name=region_name
                                           )
                self.resource = boto3.resource('s3',
                                               aws_access_key_id=access_key_id,
                                               aws_secret_access_key=secret_access_key,
                                               region_name=region_name
                                               )

            if self.bucket_name not in self.list_buckets():
                self.create_bucket(self.bucket_name)
            self.bucket = self.resource.Bucket(self.bucket_name)
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create object of AmazonSimpleStorageService in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            "__init__"))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    # def add_param(self, acceptable_param, additional_param):
    #     """
    #
    #     :param acceptable_param: specify param list can be added
    #     :param additional_param: accepts a dictionary object
    #     :return: list of param added to current instance of class
    #     """
    #     try:
    #         self.__dict__.update((k, v) for k, v in additional_param.items() if k in acceptable_param)
    #         return [k for k in additional_param.keys() if k in acceptable_param]
    #     except Exception as e:
    #         aws_exception = AWSException(
    #             "Failed to add parameter in object in module [{0}] class [{1}] method [{2}]"
    #                 .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
    #                         self.add_param.__name__))
    #         raise Exception(aws_exception.error_message_detail(str(e), sys)) from e
    #
    # def filter_param(self, acceptable_param, additional_param):
    #     """
    #
    #     :param acceptable_param: specify param list can be added
    #     :param additional_param: accepts a dictionary object
    #     :return: dict of param after filter
    #     """
    #     try:
    #         accepted_param = {}
    #         accepted_param.update((k, v) for k, v in additional_param.items() if k in acceptable_param)
    #         return accepted_param
    #     except Exception as e:
    #         aws_exception = AWSException(
    #             "Failed to filter parameter in object in module [{0}] class [{1}] method [{2}]"
    #                 .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
    #                         self.filter_param.__name__))
    #         raise Exception(aws_exception.error_message_detail(str(e), sys)) from e
    #
    # def remove_param(self, param):
    #     """
    #
    #     :param param: list of param argument need to deleted from instance object
    #     :return True if deleted successfully else false:
    #     """
    #     try:
    #         for key in param:
    #             self.__dict__.pop(key)
    #         return True
    #
    #     except Exception as e:
    #         aws_exception = AWSException(
    #             "Failed to remove parameter in object in module [{0}] class [{1}] method [{2}]"
    #                 .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
    #                         self.remove_param.__name__))
    #         raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def list_directory(self, directory_full_path=None):
        """
        :param directory_full_path:s3 directory path
        :return:
        {'status': True/False, 'message': 'message detail'
                    , 'directory_list': directory_list will be added if status is True}
        """
        try:
            if directory_full_path == "" or directory_full_path == "/" or directory_full_path is None:
                directory_full_path = ""
            else:
                if directory_full_path[-1] != "/":
                    directory_full_path += "/"

            is_directory_exist = False
            directory_list = []
            for key in self.bucket.objects.filter(Prefix=directory_full_path):
                is_directory_exist = True
                dir_name=str(key.key).replace(directory_full_path,"")
                slash_index=dir_name.find("/")
                if slash_index>=0:
                    name_after_slash=dir_name[slash_index+1:]
                    if len(name_after_slash)<=0:
                        directory_list.append(dir_name)
                else:
                    if dir_name!="":
                        directory_list.append(dir_name)

            if is_directory_exist:
                return {'status': True, 'message': 'Directory [{0}]  exist'.format(directory_full_path)
                    , 'directory_list': directory_list}
            else:
                return {'status': False, 'message': 'Directory [{0}] does not exist'.format(directory_full_path)}


        except Exception as e:
            aws_exception = AWSException(
                "Failed to list directory in object in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.list_directory.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def list_files(self, directory_full_path):
        """

        :param directory_full_path: s3 directory path name
        :return:
        {'status': True/False, 'message': 'message detail'
                    , 'files_list': files_list will be added if status is True}
        """
        try:
            if directory_full_path == "" or directory_full_path == "/" or directory_full_path is None:
                directory_full_path = ""
            else:
                if directory_full_path[-1] != "/":
                    directory_full_path += "/"

            is_directory_exist = False
            list_files = []
            for key in self.bucket.objects.filter(Prefix=directory_full_path):
                is_directory_exist = True
                file_name = str(key.key).replace(directory_full_path, "")
                if "/" not in file_name and file_name != "":
                    list_files.append(file_name)
            if is_directory_exist:
                return {'status': True, 'message': 'Directory [{0}]  present'.format(directory_full_path)
                    , 'files_list': list_files}
            else:
                return {'status': False, 'message': 'Directory [{0}] is not present'.format(directory_full_path)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to list files in object in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.list_files.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def list_buckets(self):
        """

        :return: All bucket names available in your amazon s3 bucket
        """
        try:
            response = self.client.list_buckets()
            bucket_names = [bucket_name['Name'] for bucket_name in response['Buckets']]
            return {'status': True, 'message': 'Bucket retrived', 'bucket_list': bucket_names}

        except Exception as e:
            aws_exception = AWSException(
                "Failed to list bucket in object in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.list_buckets.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def create_bucket(self, bucket_name, over_write=False):
        """

        :param bucket_name: Name of bucket
        :param over_write: If true then existing bucket content will be removed
        :return: True if created else False
        """
        try:
            bucket_list = self.list_buckets()
            if bucket_name not in bucket_list:
                self.client.create_bucket(Bucket=bucket_name)
                return True
            elif over_write and bucket_name in bucket_list:
                self.remove_directory("")
                return True
            else:
                raise Exception("Bucket [{0}] is already present".format(bucket_name))
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create bucket in object in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.create_bucket.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def create_directory(self, directory_full_path, over_write=False, **kwargs):
        """

        :param directory_full_path: provide full directory path along with name
        :param over_write: default False if accept True then overwrite existing directory if exist
        :return {'status': True/False, 'message': 'message detail'}
        """
        try:
            if directory_full_path == "" or directory_full_path == "/" or directory_full_path is None:
                return {'status': False, 'message': 'Provide directory name'}
            directory_full_path=self.update_directory_full_path_string(directory_full_path)
            response = self.list_directory(directory_full_path)

            if over_write and response['status']:
                self.remove_directory(directory_full_path)
            if not over_write:
                if response['status']:
                    return {'status': False, 'message': 'Directory is already present. try with overwrite option.'}

            possible_directory = directory_full_path[:-1].split("/")

            directory_name = ""
            for dir_name in possible_directory:
                directory_name += dir_name + "/"
                response = self.list_directory(directory_name)
                if not response['status']:
                    self.client.put_object(Bucket=self.bucket_name, Key=directory_name)
            return {'status': True, 'message': 'Directory [{0}] created successfully '.format(directory_full_path)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create directory in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.create_directory.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def remove_directory(self, directory_full_path):
        """

        :param directory_full_path:provide full directory path along with name
        kindly provide "" or "/" to remove all directory and file from bucket.
        :return:  {'status': True/False, 'message': 'message detail'}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            is_directory_found = False
            prefix_file_name = directory_full_path
            for key in self.bucket.objects.filter(Prefix=prefix_file_name):
                is_directory_found = True
                key.delete()
            if is_directory_found:
                return {'status': True, 'message': 'Directory [{0}] removed.'.format(directory_full_path)}
            else:
                return {'status': False, 'message': 'Directory [{0}] is not present.'.format(directory_full_path)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to delete directory in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.remove_directory.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def is_file_present(self, directory_full_path, file_name):
        """

        :param directory_full_path:directory_full_path
        :param file_name: Name of file
        :return: {'status': True/False, 'message': 'message detail'}
        """
        try:
            directory_full_path=self.update_directory_full_path_string(directory_full_path)
            response = self.list_files(directory_full_path)
            if response['status']:
                if file_name in response['files_list']:
                    return {'status': True, 'message': 'File [{0}] is present.'.format(directory_full_path + file_name)}
            return {'status': False, 'message': 'File [{0}] is not present.'.format(directory_full_path + file_name)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to delete directory in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.is_file_present.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def is_directory_present(self, directory_full_path):
        """

        :param directory_full_path: directory path
        :return: {'status': True/False, 'message': 'message detail"}
        """
        try:
            directory_full_path=self.update_directory_full_path_string(directory_full_path)
            response = self.list_directory(directory_full_path)
            if response['status']:
                return {'status': True, 'message': 'Directory [{0}] is present'.format(directory_full_path)}
            return {'status': False, 'message': 'Directory [{0}] is not present'.format(directory_full_path)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to delete directory in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.is_file_present.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def upload_file(self, directory_full_path, file_name,stream_data,local_file_path=False, over_write=False):
        """

        :param directory_full_path: s3 bucket directory
        :param file_name: name you want to specify for file in s3 bucket
        param stream_data: name you want to specify for file in s3 bucket
        :param local_file_path: your local system file path of file needs to be uploaded
        :param over_write:True if wanted to replace target file if present
        :return:{'status': True/False,
                    'message': 'message detail'}
        """
        try:
            if directory_full_path == "" or directory_full_path == "/":
                directory_full_path = ""
            else:
                if directory_full_path[-1] != "/":
                    directory_full_path += "/"
            response = self.is_directory_present(directory_full_path)
            if not response['status']:
                response = self.create_directory(directory_full_path)
                if not response['status']:
                    return response
            response = self.is_file_present(directory_full_path, file_name)
            if response['status'] and not over_write:
                return {'status': False,
                        'message': 'File [{0}] already present in directory [{1}]. try with overwrite option'.format(
                            file_name, directory_full_path)}
            if local_file_path:
                self.bucket.upload_file(Filename=local_file_path, Key=directory_full_path + file_name)
            else:
                if isinstance(stream_data,str):
                    stream_data=io.StringIO(stream_data)
                self.client.put_object(Bucket=self.bucket_name, Key=directory_full_path + file_name, Body=stream_data)
            return {'status': True,
                    'message': 'File [{0}] uploaded to directory [{1}]'.format(file_name, directory_full_path)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to upload file in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.upload_file.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def download_file(self, directory_full_path, file_name, local_system_directory=""):
        """

        :param directory_full_path:directory_full_path
        :param file_name: Name of file
        :param local_system_directory: file location within your system
        :return: {'status': True/False,
                    'message':'message detail'}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path=directory_full_path, file_name=file_name)
            local_system_directory = self.update_directory_full_path_string(local_system_directory)
            if not response['status']:
                return response
            self.client.download_file(self.bucket_name, directory_full_path + file_name,
                                      local_system_directory + file_name)
            return {'status': True,
                    'message': 'file [{0}] is downloaded in your system at location [{1}] '
                        .format(file_name, local_system_directory)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to upload file in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.download_file.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def remove_file(self, directory_full_path, file_name):
        """
        :param directory_full_path: provide full directory path along with name
        :param file_name: file name with extension if possible
        :return: {'status': True/False,
                    'message':'message detail'}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if response['status']:
                self.resource.Object(self.bucket_name, directory_full_path + file_name).delete()
                return {'status': True,
                        'message': 'File [{}] deleted from directory [{}]'.format(file_name,directory_full_path)}
            return {'status': False, 'message': response['message']}

        except Exception as e:
            aws_exception = AWSException(
                "Failed to remove file in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.remove_file.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def write_file_content(self, directory_full_path, file_name, content, over_write=False):
        """

        :param directory_full_path:  provide full directory path along with name
        :param file_name: file name with extension if possible
        :param content: content need to store in file
        :param over_write:  default False if accept True then overwrite file in directory if exist
        :return: {'status': True/False,
                    'message':'message detail'}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_directory_present(directory_full_path)
            if not response['status']:
                response = self.create_directory(directory_full_path)
                if not response['status']:
                    return {'status': False,
                            'message': 'Failed to created directory [{0}] [{1}]'.format(directory_full_path,
                                                                                        response['message'])}
            response = self.is_file_present(directory_full_path, file_name)
            if response['status'] and not over_write:
                return {'status': False,
                        "message": "File [{0}] is already present in directory [{1}]. try with over write option".format(
                            file_name, directory_full_path)}

            self.client.upload_fileobj(io.BytesIO(dill.dumps(content)), self.bucket_name,
                                       directory_full_path + file_name)
            return {'status': True,
                    'message': 'File [{0}] is created in directory [{1}]'.format(file_name, directory_full_path)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.write_file_content.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def update_directory_full_path_string(self, directory_full_path):
        """

        :param directory_full_path: directory_full_path
        :return: update the accepted directory
        """
        try:
            if directory_full_path == "" or directory_full_path == "/":
                directory_full_path = ""
            else:
                if directory_full_path[-1] != "/":
                    directory_full_path = directory_full_path + "/"
            return directory_full_path
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.update_directory_full_path_string.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def read_csv_file(self, directory_full_path, file_name):
        """

        :param directory_full_path: directory_full_path
        :param file_name: file_name
        :return: {'status': True/False,
                    'message': 'message detail',
                    'data_frame': if status True data frame will be returned}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if not response['status']:
                return response
            content = io.BytesIO()
            self.client.download_fileobj(self.bucket_name, directory_full_path + file_name, content)
            content.seek(0)
            df = pd.read_csv(content,header= 0,
                     index_col= False)
            return {'status': True,
                    'message': 'File [{0}] has been read into data frame'.format(directory_full_path + file_name),
                    'data_frame': df}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.update_directory_full_path_string.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e


    def read_json_file(self, directory_full_path, file_name):
        """

        :param directory_full_path: s3 bucket directory_full_path
        :param file_name: file_name
        :return: {'status': True/False, 'message': 'message_detail',
                    'file_content':'If status True then Return object which was used to generate the file with write file content'}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if not response['status']:
                return response
            content = io.BytesIO()
            self.client.download_fileobj(self.bucket_name, directory_full_path + file_name, content)
            return {'status': True, 'message': 'File [{0}] has been read'.format(directory_full_path + file_name),
                    'file_content': json.loads(content.getvalue())}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to reading json content in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.read_json_file.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def read_file_content(self, directory_full_path, file_name):
        """

        :param directory_full_path: directory_full_path
        :param file_name: file_name
        :return: {'status': True/False, 'message': 'message_detail',
                    'file_content':'If status True then Return object which was used to generate the file with write file content'}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if not response['status']:
                return response
            content = io.BytesIO()
            self.client.download_fileobj(self.bucket_name, directory_full_path + file_name, content)
            return {'status': True, 'message': 'File [{0}] has been read'.format(directory_full_path + file_name),
                    'file_content': dill.loads(content.getvalue())}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.read_file_content.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e
    def get_pickle_file(self, directory_full_path, file_name):
        """

        :param directory_full_path: directory_full_path
        :param file_name: file_name
        :return: {'status': True/False, 'message': 'message_detail',
                    'file_content':'If status True then Return object which was used to generate the file with write file content'}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if not response['status']:
                return response
            # content = io.BytesIO()
            # self.client.download_fileobj(self.bucket_name, directory_full_path + file_name, content)
            with BytesIO() as data:
                self.resource.Bucket(self.bucket_name).download_fileobj(directory_full_path + file_name, data)
                data.seek(0)
                # return pickle.load(data)
                # scaler = pickle.load(data)
                return {'status': True, 'message': 'File [{0}] has been read'.format(directory_full_path + file_name),
                    'file_content': pickle.load(data)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.read_file_content.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e
    def move_file(self, source_directory_full_path, target_directory_full_path, file_name, over_write=False,
                  bucket_name=None):
        """

        :param source_directory_full_path: provide source directory path along with name
        :param target_directory_full_path: provide target directory path along with name
        :param file_name: file need to move
        :param over_write:  default False if accept True then overwrite file in target directory if exist
        :return: {'status': True/False,
                        'message': 'message detail'}
        """
        try:
            response = self.copy_file(source_directory_full_path, target_directory_full_path, file_name, over_write,
                                      bucket_name)
            if not response['status']:
                return {'status': False, 'message': 'Failed to move file due to [{}]'.format(response['message'])}
            else:
                if bucket_name is None:
                    bucket_name = self.bucket_name
                self.remove_file(source_directory_full_path, file_name)
                return {'status': True,
                        'message': 'File moved successfully from bucket: [{0}] directory [{1}] to bucket:[{2}] '
                                   'directory[{3}]'.format(self.bucket_name,
                                                           source_directory_full_path + file_name, bucket_name,
                                                           target_directory_full_path + file_name)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.move_file.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

    def get_current_bucket_name(self):
        """
        :return: The current bucket name which is in use.
        """
        return self.bucket_name
    def change_current_bucket(self,bucket_name):
        """

        :param bucket_name: Provide the bucket_name which needs to be changed for further use
        :return: Change the current bucket if it exists and return the message
        """
        if bucket_name not in self.list_buckets()['bucket_list']:
            return f"{bucket_name} does not exists.You need to create bucket first"
        elif self.bucket_name==bucket_name:
            return f"{bucket_name} is already in use"
        else:
            self.bucket_name=bucket_name
            return f"Now Bucket->{self.bucket_name} is in use."
    def copy_file(self, source_directory_full_path, target_directory_full_path, file_name, over_write=False,
                  bucket_name=None):
        """

        :param source_directory_full_path: provide source directory path along with name
        :param target_directory_full_path: provide target directory path along with name
        :param file_name: file need to copy
        :param over_write: default False if accept True then overwrite file in target directory if exist
        :return: {'status': True/False,
                        'message': 'message detail'}
        """
        try:
            target_directory_full_path = self.update_directory_full_path_string(target_directory_full_path)
            source_directory_full_path = self.update_directory_full_path_string(source_directory_full_path)
            response = self.is_file_present(source_directory_full_path, file_name)
            if not response['status']:
                return {'status': False,
                        'message': 'Source file [{0}] is not present'.format(source_directory_full_path + file_name)}
            if bucket_name is None:
                bucket_name = self.bucket_name
                aws_obj = self
            else:
                bucket_name = bucket_name
                aws_obj = AmazonSimpleStorageService(bucket_name=bucket_name)

            response = aws_obj.is_file_present(target_directory_full_path, file_name)
            if response['status'] and not over_write:
                return {'status': False,
                        'message': 'Bucket:[{0}] target directory '
                                   '[{1}] contains file [{2}] please'
                                   ' try with over write option.'.format(bucket_name,
                                                                         target_directory_full_path,
                                                                         file_name
                                                                         )}

            copy_source = {
                'Bucket': self.bucket_name,
                'Key': source_directory_full_path + file_name
            }
            response = aws_obj.is_directory_present(target_directory_full_path)

            if not response['status']:
                response = aws_obj.create_directory(target_directory_full_path)
                if not response['status']:
                    return {'status': False,
                            'message': 'Failed to created'
                                       ' target directory [{}] in bucket:[{}]'.format(
                                target_directory_full_path,
                                bucket_name
                            )}

            self.client.copy(copy_source, bucket_name, target_directory_full_path + file_name)
            return {'status': True,
                    'message': 'File copied successfully from bucket: [{0}] directory [{1}] to bucket:[{2}] '
                               'directory[{3}]'.format(self.bucket_name,
                                                       source_directory_full_path + file_name, bucket_name,
                                                       target_directory_full_path + file_name)}
        except Exception as e:
            aws_exception = AWSException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(AmazonSimpleStorageService.__module__.__str__(), AmazonSimpleStorageService.__name__,
                            self.copy_file.__name__))
            raise Exception(aws_exception.error_message_detail(str(e), sys)) from e

def download_data():
    df=pd.read_csv(r"https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv",skiprows=range(0,20)).to_csv('training_data.csv',index=False)
    return r'./training_data.csv'
if __name__=='__main__':
    # aws=AmazonSimpleStorageService('AKIAXZVACUKMCTAMGNL7','ccp+kXF+vv3NycDr3jGUr8jk345ZpQTbsxBsYcst','scania-121')
    # aws=AmazonSimpleStorageService('AKIAXZVACUKMLFZWYAOI','DI+XyK3qlJX3qzUXfUke2q+uJQamLExKdjQGCefu','machine-learning-6652')
    aws=AmazonSimpleStorageService('AKIAXZVACUKMP6ZTKK3E','veUIBpOnwohAe5Wc5pGzCMtJfT0u+fsFZFlQsFAc','machine-learning-6652')

    # print(aws.create_directory('artifacts'))
    # print(aws.is_directory_present('test1'))
    # print(aws.upload_file(r'company-name/','training_data.csv','training_data.csv',local_file_path='training_data.csv'))
    # print('hello')
    # aws.read_file_content('company-name','google_cloud_storage.py')
    # df=aws.read_csv_file(r'company-name/','training_data.csv')['data_frame']
    # print(df)
    # print(aws.list_buckets())

    # print(aws.download_file('company-name','training_data.csv',r'D:\CloudStorageAutomation\exception_layer'))
    # print(aws.get_current_bucket_name())
    # print(aws.change_current_bucket('machine-learning-6652'))
    # print(aws.list_buckets())