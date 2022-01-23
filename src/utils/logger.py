from datetime import datetime
import os

class App_Logger:

    """
    This class Shall be used to implement the logging locally
    author : Ankit Sharma

    """
    def __init__(self):
        self.path="src/logs"
    def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        if os.path.exists(os.path.join(self.path,f"{file_object}.log")):
            self.file_object=open((os.path.join(self.path,f"{file_object}.log")),'a')
        else:
            self.file_object = open((os.path.join(self.path, f"{file_object}.log")), 'w')
        self.file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")
