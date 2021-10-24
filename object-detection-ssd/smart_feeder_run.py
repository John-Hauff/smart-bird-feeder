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
import serial
import cv2
import numpy as np

import jetson.inference
import jetson.utils

import argparse
import sys

# http post request img file in req body
from send_img import post_bird_memory
# push notifications
from push_notification import send_push_message
# import emailing capabilities
import emailer


def serial_config():
    print("UART Demo Program to signal ML net + camera to start")
    print("NVIDIA Jetson Nano Developer Kit")

    serial_port = serial.Serial(
        port="/dev/ttyTHS1",
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )
    # Wait a second to let the port initialize
    time.sleep(1)

    return serial_port


# Function to update title bar of capture window
def update_title_bar(output, title):
    output.SetStatus(title)


def squirrel_check(net, detections):
    # check if a squirrel was detected in the frame
    for detection in detections:
        if str(net.GetClassDesc(detection.ClassID)) == 'squirrel' and detection.Confidence >= .50:
            return True


def handle_squirrel():
    print('closing hatch')
    close_hatch_cmd = 'c'
    # write msg to UART serial port
    serial_port.write(close_hatch_cmd.encode())


def handle_bird(net, detections, species_names, img, counter, species_to_ignore):
    print('opening hatch')
    open_hatch_cmd = 'o'
    # write msg to UART serial port
    serial_port.write(open_hatch_cmd.encode())

    # this loop works only when an object (or objects) is detected
    for detection in detections:
        species_label = str(net.GetClassDesc(detection.ClassID))

        if counter == 150:
            counter = 0
            # set species occurence vals False so that we can detect them again.
            # species_has_occured[species_to_ignore] = False
            species_to_ignore = species_label
            # species_has_occured = {species_label: False for species_label in species_has_occured}
        else:
            # Increment counter
            counter += 1

        if detection.Confidence >= 0.90 and species_to_ignore != species_label:
            print(detection)
            print('processing species: ' +
                  str(net.GetClassDesc(detection.ClassID)))
            ## handle confidently detected bird ##
            # save capturedAt time (may not use)
            timestamp = str(time.time())
            # save_img(img, timestamp)
            # post saved bird memory with formatted species name
            # post_bird_memory(
            #    species_names[species_label])
            # send push notification for newly added bird memory
            token = 'ExponentPushToken[QdzwK-NUMCWMaVSyKnb8BC]'
            title = 'New Bird Memory! 🐦'
            message = 'A new bird memory has been captured!\nView it in your bird memories gallery.'
            # send_push_message(token, title, message)
            # emailer.send_bird_memory(
            #    net, detection, img, timestamp)
    return species_to_ignore


# Function that will write the current frame as a .jpg to local storage
def save_img(img, timestamp):
    cv2.imwrite("captured-bird-images/" + str('bird_memory' + ".jpeg"),
                cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    # parse the command line
    parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                     jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

    parser.add_argument("input_URI", type=str, default="",
                        nargs='?', help="URI of the input stream")
    parser.add_argument("output_URI", type=str, default="",
                        nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                        help="pre-trained model to load (see below for options)")
    parser.add_argument("--overlay", type=str, default="none",
                        help="detection overlay flags (e.g. --overlay=labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="minimum detection threshold to use")

    is_headless = [
        "--headless"] if sys.argv[0].find('console.py') != -1 else [""]

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
    output = jetson.utils.videoOutput(
        opt.output_URI, argv=sys.argv+is_headless)

    # setup serial communication
    serial_port = serial_config()

    # dict for bird species labels to formatted names
    species_names = {
        'american-crow': 'american crow',
        'blue-jay': 'blue jay',
        'blue-gray-gnatcatcher': 'blue gray gnatcatcher',
        'carolina-wren': 'carolina wren',
        'common-grackle': 'common grackle',
        'downy-woodpecker': 'downy woodpecker',
        'gray-catbird': 'gray catbird',
        'mourning-dove': 'mourning dove',
        'cardinal': 'northern cardinal',
        'northern-mockingbird': 'northern mockingbird',
        'palm-warbler': 'palm warbler',
        'pileated-woodpecker': 'pileated woodpecker',
        'red-bellied-woodpecker': 'red-bellied woodpecker',
        'tufted-titmouse': 'tufted titmouse',
        'yellow-rumped-warbler': 'yellow-rumped warbler',
        'squirrel': 'squirrel'
    }

    species_has_occured = {
        'american-crow': False,
        'blue-jay': False,
        'blue-gray-gnatcatcher': False,
        'carolina-wren': False,
        'common-grackle': False,
        'downy-woodpecker': False,
        'gray-catbird': False,
        'mourning-dove': False,
        'cardinal': False,
        'northern-mockingbird': False,
        'palm-warbler': False,
        'pileated-woodpecker': False,
        'red-bellied-woodpecker': False,
        'tufted-titmouse': False,
        'yellow-rumped-warbler': False,
        'squirrel': False
    }

    counter = 0
    # Set an arbitrary bird species to start off with.
    # This keeps track of what the last bird was.
    species_to_ignore = 'american-crow'

    try:
        # Send a simple header
        serial_port.write(
            "UART Demo Program to signal ML net + camera to start\r\n".encode())
        serial_port.write("NVIDIA Jetson Nano Developer Kit\r\n".encode())

        while True:
            if serial_port.in_waiting > 0:
                data = serial_port.read()
                print(data)
                serial_port.write(data)

                if data == "\r".encode():
                    # For Windows boxen on the other end
                    serial_port.write("\n".encode())

                if data == 'r'.encode():
                    print('r received!')
                    serial_port.write('a'.encode())  # ack msg

                    while True:
                        # read serial port for stop message (received when MCU's sensor stops detecting objects)
                        if serial_port.in_waiting > 0 and serial_port.read() == 's'.encode():
                            serial_port.write('a'.encode())  # ack msg
                            break
                        else:
                            ################################# object detection code #################################
                            # process frames until the user exits
                            # capture the next image
                            img = input.Capture()

                            # detect objects in the image (with overlay chosen in parser arguments)
                            detections = net.Detect(img, overlay=opt.overlay)

                            # print the detections
                            print("detected {:d} object(s) in image".format(
                                len(detections)))

                            # render the image
                            output.Render(img)

                            # update the title bar
                            update_title_bar(output, "{:s} | Network {:.0f} FPS".format(
                                opt.network, net.GetNetworkFPS()))

                            squirrel_detected = False

                            # check if a squirrel was detected in the frame
                            squirrel_detected = squirrel_check(net, detections)

                            if squirrel_detected:
                                print('squirrel detected!')  # debug
                                ## handle squirrel prescence ##
                                handle_squirrel(serial_port)
                                break  # stop processing current frame

                            if not squirrel_detected:
                                species_to_ignore = handle_bird(
                                    net, detections, species_names, img, counter, species_to_ignore)

                            # print out performance info
                            net.PrintProfilerTimes()

                            # exit on input/output EOS
                            if not input.IsStreaming() or not output.IsStreaming():
                                break

    except KeyboardInterrupt:
        print("Exiting Program")

    except Exception as exception_error:
        print("Error occurred. Exiting Program")
        print("Error: " + str(exception_error))

    finally:
        serial_port.close()
        pass



