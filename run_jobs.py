import json
import os
import subprocess
import tempfile

from absl import app
from absl import logging

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

try:
    import boto3
except ImportError:
    boto3 = None

_CONFIG_JSON = "config.json"


def read_config():
    with open(_CONFIG_JSON) as f:
        return json.load(f)


def create_multipart_message(
        sender: str, recipients: list, title: str, text: str=None, html: str=None, attachments: list=None)\
        -> MIMEMultipart:
    """
    Creates a MIME multipart message object.
    Uses only the Python `email` standard library.
    Emails, both sender and recipients, can be just the email string or have the format 'The Name <the_email@host.com>'.

    :param sender: The sender.
    :param recipients: List of recipients. Needs to be a list, even if only one recipient.
    :param title: The title of the email.
    :param text: The text version of the email body (optional).
    :param html: The html version of the email body (optional).
    :param attachments: List of files to attach in the email.
    :return: A `MIMEMultipart` to be used to send the email.
    """
    multipart_content_subtype = 'alternative' if text and html else 'mixed'
    msg = MIMEMultipart(multipart_content_subtype)
    msg['Subject'] = title
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    # Record the MIME types of both parts - text/plain and text/html.
    # According to RFC 2046, the last part of a multipart message, in this case the HTML message, is best and preferred.
    if text:
        part = MIMEText(text, 'plain')
        msg.attach(part)
    if html:
        part = MIMEText(html, 'html')
        msg.attach(part)

    # Add attachments
    for attachment in attachments or []:
        with open(attachment, 'rb') as f:
            part = MIMEApplication(f.read())
            part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment))
            msg.attach(part)

    return msg


def send_mail(
        sender: str, recipients: list, title: str, text: str=None, html: str=None, attachments: list=None, region_name=None) -> dict:
    """
    Send email to recipients. Sends one mail to all recipients.
    The sender needs to be a verified email in SES.
    """
    msg = create_multipart_message(sender, recipients, title, text, html, attachments)
    ses_client = boto3.client('ses', region_name=region_name)
    return ses_client.send_raw_email(
        Source=sender,
        Destinations=recipients,
        RawMessage={'Data': msg.as_string()}
    )


def main(argv):
    del argv

    config = read_config()
    sender = config["sender"]
    recipient = config["recipient"]
    region_name = config.get("region_name", None)

    with open('jobs.txt', 'r') as f:
        for job in f.readlines():
            job = job.strip()
            if not job:
                continue

            proc = subprocess.Popen(
                job.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            outs, _ = proc.communicate()
            returncode = proc.returncode

            if returncode:
                logging.warning('[FAILURE] %s:\n %s', job, outs)
                subject = f"task {job} failed with return code {returncode}"
                with tempfile.NamedTemporaryFile(suffix=".txt") as fp:
                    fp.write(outs)
                    fp.flush()

                    try:
                        send_mail(
                            sender=sender,
                            recipients=[recipient],
                            title=subject,
                            text="",
                            attachments=[fp.name],
                            region_name=region_name)
                    except:
                        logging.warning("[WARNING] could not send message from %s to %s in region %s.",
                                        sender, recipient, region_name)
            else:
                logging.info('[SUCCESS] %s', job)

if __name__ == "__main__":
    app.run(main)
