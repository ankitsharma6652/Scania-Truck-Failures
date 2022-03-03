import smtplib
from cryptography.fernet import Fernet
from email.message import EmailMessage
import smtplib
import requests


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

    def validate_email(self,email_address):
        try:
            response = requests.get(
                "https://isitarealemail.com/api/email/validate",
                params={'email': email_address})

            status = response.json()['status']
            print(status)
            if status == "valid":
                print("email is valid")
                return True
            elif status == "invalid":
                print("email is invalid")
                return False
            else:
                print("email was unknown")
                return False
        except Exception as e:
            print(e)
            return False

    def notify_email(self,email):
        try:
            message = EmailMessage()
            message["Subject"] = "Scania Truck Failures Training Notification"
            message["From"] = "digitalaks9@gmail.com"
            message["To"] = email
            text = f"Hi recipient,\n\n This is notification email from Scania Truck Failures Application.\n\n" \
                   f"Description: \n\n We are going to Start Model Training.We will notify you once the training gets completed. \n\n Thanks & Regards," \
                   f"\nAnkit Sharma"
            message.set_content(text)
            # Create secure connection with server and send email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login("digitalaks9@gmail.com", "erdceuisbtejdrtc")
                smtp.send_message(message)
            return True

        except Exception as e:
            print(e)
            # raise Exception(e)
            return False

    def send_email(self,mail_text,TO):
        """
        message: Message string in html format
        subject: subject of email
        """
        try:
            message = EmailMessage()
            message["Subject"] = "Scania Truck Failures Training Completed"
            message["From"] = "digitalaks9@gmail.com"
            message["To"] = TO
            text = f"Hi recipient,\n\n This is notification email from Scania Truck Failures Application.\n\n" \
                   f"Description: \n\n{mail_text} \n\n Thanks & Regards," \
                   f"\nAnkit Sharma"
            message.set_content(text)
            # Create secure connection with server and send email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login("digitalaks9@gmail.com", "erdceuisbtejdrtc")
                smtp.send_message(message)
        except Exception as e:
            print(e)
            raise Exception(e)
