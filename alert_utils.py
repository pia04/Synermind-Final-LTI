import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List

def send_alert(subject: str, body: str, recipients: List[str], smtp_server: str, smtp_port: int, sender_email: str, sender_password: str):
    """
    Send an alert email (or SMS via email-to-SMS gateway) to the specified recipients.
    :param subject: Email subject
    :param body: Email body
    :param recipients: List of recipient email addresses (can include email-to-SMS addresses)
    :param smtp_server: SMTP server address (e.g., 'smtp.gmail.com')
    :param smtp_port: SMTP port (e.g., 587)
    :param sender_email: Sender's email address
    :param sender_password: Sender's email password or app password
    """
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        print(f"Alert sent to: {recipients}")
    except Exception as e:
        print(f"Failed to send alert: {e}")
