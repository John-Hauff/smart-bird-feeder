#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import time as t
import cv2
import numpy as np

import smtplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

import jetson.inference
import jetson.utils

import argparse
import sys

# Function to update title bar of capture window
def update_title_bar(title):
	output.SetStatus(title)


# Function that will write the current frame as a .jpg to local storage
# The img name is the detected class description and a unique time stamp ID
def save_img(img, timestamp):
	cv2.imwrite("captured-bird-images/" +
	str(net.GetClassDesc(detection.ClassID)) +
	"_" + str(timestamp + ".jpg", cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)))
 

# TODO: Clean up and make sense out of this send email function (ask Paul)
# Also - probably want to move this function to antoher file and import it instead (this would help avoid merge conflicts as well)
def send_email(img, timestamp):
	# ------------------------------------Email photos----------------------------------------------
	smtp_user = "sdgroup7project@gmail.com"
	smtp_pass = "bQlh#cQLkZ%d"

	# Destination
	to_add = "matthew.a.wilkinson@gmail.com"
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

	# Attach image of the bird that got high confidence.
	fp = open('captured-bird-images-' + str(net.GetClassDesc(detection.ClassID)) + "_" + timestamp + '.jpg', 'rb')
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
  
  
if __name__ == '__main__':
	# parse the command line
	parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
																	formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
																	jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

	parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
	parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
	parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
	parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
	parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

	is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

	try:
		opt = parser.parse_known_args()[0]
	except:
		print("")
		parser.print_help()
		sys.exit(0)

	# load the object detection network
	net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

	# create video sources & outputs
	input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
	output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

	# process frames until the user exits
	while True:
		# capture the next image
		img = input.Capture()

		# detect objects in the image (with overlay chosen in parser arguments)
		detections = net.Detect(img, overlay=opt.overlay)

		# print the detections
		print("detected {:d} objects in image".format(len(detections)))

		# render the image
		output.Render(img)
		
		# update the title bar
		output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
		
		# This loop works only when an object (or objects) is detected
		for detection in detections:
			if detection.Confidence >= 0.90:
				update_title_bar("Confidence lvl is >= 90% and class is " + str(net.GetClassDesc(detection.ClassID)))

				timestamp = t.time()
				save_img(img, timestamp)
    
				# send_email(img, timestamp)
			
		# print out performance info
		net.PrintProfilerTimes()

		# exit on input/output EOS
		if not input.IsStreaming() or not output.IsStreaming():
			break


