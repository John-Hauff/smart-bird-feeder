import smtplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders


def send_bird_memory(net, detection, img, timestamp):
    # ~~~~~~~~~~~~~~~~~~ Email photos ~~~~~~~~~~~~~~~~~~ #
    smtp_user = "sdgroup7project@gmail.com"
    smtp_pass = "bQlh#cQLkZ%d"

    # Destination
    # to_add = "matthew.a.wilkinson@gmail.com"
    to_add = "jthauff@gmail.com"
    from_add = smtp_user

    subject = "Bird feeder picture " + timestamp
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = from_add
    msg["To"] = to_add

    msg.preamble = "Photos from: " + timestamp

    # Email Text
    body = MIMEText("Photos from: " + timestamp)
    msg.attach(body)

    # Attach image of the bird that got high confidence
    fp = open('captured-bird-images/bird_memory.jpeg', 'rb')

    # Old approach to bird memory filename(s):
    # fp = open("captured-bird-images/" +
    # str(net.GetClassDesc(detection.ClassID)) +
    # "_" + str(timestamp) + ".jpg", 'rb')

    img = MIMEImage(fp.read())
    fp.close()
    msg.attach(img)

    # Send Email

    # Gmail uses port 587.
    s = smtplib.SMTP("smtp.gmail.com", 587)

    # Encryption & sending email.
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(smtp_user, smtp_pass)
    s.sendmail(from_add, to_add, msg.as_string())
    s.quit()

    print("Email Sent.")
