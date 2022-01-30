# from exception_layer.generic_exception.generic_exception import GenericException as EmailSenderException
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys
# from project_library_layer.credentials import credential_data
from cryptography.fernet import Fernet
import os
from email.message import EmailMessage

import smtplib
from email.mime.text import MIMEText
import sys
# from project_library_layer.credentials import credential_data


# class EmailSender:
#
#     def __init__(self):
#         try:
#             sender_credentials = credential_data.get_sender_email_id_credentials()
#             self.__sender_email_id = sender_credentials.get('email_address', None)
#             self.__passkey = sender_credentials.get('passkey', None)
#             self.__receiver_email_id = credential_data.get_receiver_email_id_credentials()
#         except Exception as e:
#             email_sender_excep = EmailSenderException(
#                 "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
#                     .format(self.__module__, EmailSender.__name__,
#                             self.__init__.__name__))
#             raise Exception(email_sender_excep.error_message_detail(str(e), sys)) from e
def get_password():
    try:
        f = Fernet("ZbuysL2K5N_085_nF6NMxLTHApxLMMIgk3OTXQSDuQ4=")
        print(f.decrypt(b"gAAAAABh9Lg7k5ljtas5hbhwQEnIbCq7qE7N9DE18JpJnMQGiNvwdJwjqk_vkuj1XddJCe0jv0EBYbKO-8fBQssNYhu8I9GPpw==").decode())
        return f.decrypt(b"gAAAAABh9Lg7k5ljtas5hbhwQEnIbCq7qE7N9DE18JpJnMQGiNvwdJwjqk_vkuj1XddJCe0jv0EBYbKO-8fBQssNYhu8I9GPpw==").decode()
    except Exception as e:
        print(e)
        raise Exception(e)
class email_sender:
    def __init__(self):
       pass
    def set_reciever_mail(self,mail):
        self.reciever_mail = mail
    def get_reciever_mail(self):
        return  self.reciever_mail
    def send_email(self,mail_text,TO):
        """
        message: Message string in html format
        subject: subject of email
        """
        try:
            message = EmailMessage()
            message["Subject"] = "Scania Truck Failures Training Completed"
            message["From"] = "ankitcoolji@gmail.com"
            message["To"] = TO
            text = f"Hi recipient,\n\n This is notification email from Scania Truck Failures Application.\n\n" \
                   f"Description: \n\n{mail_text} \n\n Thanks & Regards," \
                   f"\nAnkit Sharma"
            message.set_content(text)
            # Create secure connection with server and send email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login("ankitcoolji@gmail.com", "ltcftqdsqssptlbu")
                smtp.send_message(message)
        except Exception as e:
            print(e)
            raise Exception(e)
            # email_sender_excep = EmailSenderException(
            #     "Failed during sending email module [{0}] class [{1}] method [{2}]"
            #         .format(self.__module__, EmailSender.__name__,
            #                 self.send_email.__name__))
            # raise Exception(email_sender_excep.error_message_detail(str(e), sys)) from e
# email=email_sender()
# email.reciever_mail="digiaks9@gmail.com"
# # print(email.get_reciever_mail())
# email.send_email("hello")
# # send_email("hello","digiaks9@gmail.com")