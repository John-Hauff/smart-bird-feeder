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

import time
import cv2
import numpy as np


import jetson.inference
import jetson.utils

import argparse
import sys

# http post request img file in req body
import send_img
# import emailing capabilities
import emailer

# Function to update title bar of capture window
def update_title_bar(output, title):
	output.SetStatus(title)

# Function that will write the current frame as a .jpg to local storage
# DEPRECATED APPROACH: The img name is the detected class description and a unique time stamp ID
# CURRENT APPROACH: The img name will always be 'bird_memory.jpeg', and each new bird memory will
# overwrite the old bird memory in local storage, which is fine as long as the image is backed up to the db
def save_img(img, timestamp):
	cv2.imwrite("captured-bird-images/" + str('bird_memory' + ".jpeg"), cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
	# str(net.GetClassDesc(detection.ClassID)) +
	# "_" + timestamp + ".jpeg", cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
	# parse the command line
	parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
																	formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
																	jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

	parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
	parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
	parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
	parser.add_argument("--overlay", type=str, default="none", help="detection overlay flags (e.g. --overlay=labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
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
		print("detected {:d} object(s) in image".format(len(detections)))

		# render the image
		output.Render(img)
		
		# update the title bar
		update_title_bar(output, "{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
		
		# This loop works only when an object (or objects) is detected
		for detection in detections:
			if detection.Confidence >= 0.90:
				print(detection)
				timestamp = str(time.time()) # save capturedAt time (may not use)
				save_img(img, timestamp)
				send_img.post_bird_memory()
				emailer.send_bird_memory(net, detection, img, timestamp)
    
			
		# print out performance info
		net.PrintProfilerTimes()

		# exit on input/output EOS
		if not input.IsStreaming() or not output.IsStreaming():
			break


