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

# token for Expo push notifications
token = 'ExponentPushToken[QdzwK-NUMCWMaVSyKnb8BC]'
# counter1 for counting interval to ignore a species for
counter1 = 0
# counter2 for waiting some frames after a squirrel is gone before opening hatch again
counter2 = 0
# counter3 for waiting some time before saving a captured bird img so that bird has time to settle for a good shot
counter3 = 0
hatch_is_open = True


def run_obj_detection(input, output, net, opt, serial_port, species_names, species_to_ignore):
    global hatch_is_open, counter2

    ################################# object detection code #################################
    # capture the next image
    img = input.Capture()

    # copy img to preserve no overlay in img but still have an overlayed img to render to ouput window
    # this prevents detection overlay from showing up in bird memories
    overlayed_img = img

    # detect objects in the image (with overlay chosen in parser arguments)
    detections = net.Detect(overlayed_img, overlay=opt.overlay)

    # print the detections
#    print("detected {:d} object(s) in image".format(len(detections)))

    # render the image with detections overlay
    output.Render(overlayed_img)

    # update the title bar
    update_title_bar(output, "{:s} | Network {:.0f} FPS".format(
        opt.network, net.GetNetworkFPS()))

    squirrel_detected = False

    # check if a squirrel was detected in the frame
    squirrel_detected = is_squirrel_detected(net, detections)

    if squirrel_detected:
        counter2 = 0
        ## handle squirrel prescence ##
        handle_squirrel(serial_port)
        return  # stop processing current frame
    else:
        counter2 += 1

        # if hatch closed and squirrel was not detected for a while -> open hatch and handle bird detection
        if not hatch_is_open and counter2 >= (30 * 5):
            counter2 = 0
            # opening hatch also causes the alarm to stop sounding
            open_hatch(serial_port)
            hatch_is_open = True
        species_to_ignore = handle_bird(
            net, detections, species_names, img, species_to_ignore)

    # print out performance info
#    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        return species_to_ignore

    return species_to_ignore


def is_squirrel_detected(net, detections):
    # check if a squirrel was detected in the frame
    for detection in detections:
        if str(net.GetClassDesc(detection.ClassID)) == 'squirrel' and detection.Confidence >= .90:
            return True


def handle_squirrel(serial_port):
    print('Start of handle_squirrel')
    global hatch_is_open
    
    if hatch_is_open:
        # closing hatch also causes the alarm to start sounding until hatch is opened again
        close_hatch(serial_port)
        hatch_is_open = False


def handle_bird(net, detections, species_names, img, species_to_ignore):
    global counter1, counter3

    # this loop works only when an object (or objects) is detected
    for detection in detections:
        species_label = str(net.GetClassDesc(detection.ClassID))

        if counter1 >= (30 * 5):
            print('counter1 has reached {:d}. Now resetting counter1 and species_to_ignore'.format(counter1))
            counter1 = 0
            # reset species to ignore to None with the current detected species
            species_to_ignore = None
        else:
            counter1 += 1
            counter3 += 1

        if detection.Confidence >= 0.90 and species_to_ignore != species_label:
#            print(detection)
            print('Processing species:', species_label)  # debug

            # reassign species to ignore with the current detected species
            species_to_ignore = species_label
            # reset counter1 so full time is waited before repeating capture of ignored species
            counter1 = 0

            ## handle confidently detected bird ##
            # save capturedAt time (may not use)
            # Wait some time before saving bird img for non-ignored species
            # to allow time for bird to settle and prevent blurry saved imgs.
            # After the first img of a newly detected species is saved, if the same species is repeatedly detected,
            # without being interrupted by detections of other species, this block will only execute when counter1 peaks
            # and the ignored species is reset, making counter3 temporarily useless.
            # TODO: fix up counter3 implementation. It's not perfect. Will counter3 reset if a new species is detected before counter3 peaks?
            if counter3 >= (30 * 3):
                print('counter3 has reached {:d}. Now resetting counter3 and saving detected bird img'.format(counter3))
                counter3 = 0
                # TODO: maybe use this time stamp for memory creation time instead of mongoose Date object...
                timestamp = str(time.time())
#                save_img(img, timestamp)
                # post saved bird memory with formatted species name
#                post_bird_memory(species_names[species_label])
                # send push notification for newly added bird memory
                title = 'New Bird Memory! 🐦'
                message = 'A new bird memory has been captured!\nView it in your bird memories gallery.'
#                send_push_message(token, title, message)

    return species_to_ignore

def should_check_feed_lvl(time1, time2):
    # TODO: adjust wait time for low feed check
    wait_time = 30  # waiting interval in seconds
    return (time2 - time1) >= wait_time


def open_hatch(serial_port):
    print('Start of open_hatch function')
    open_hatch_cmd = 'o'
    # write msg to UART serial port
    if serial_port.in_waiting > 0:
        print('Data found in serial port in check #1') # debug
        data = serial_port.read()
        handle_serial_data(data)
    serial_port.write(open_hatch_cmd.encode())
    print('Hatch open command sent to MSP430')


def close_hatch(serial_port):
    print('Start of close_hatch function')
    close_hatch_cmd = 'c'
    if serial_port.in_waiting > 0:
        print('Data found in serial port in check #2') # debug
        data = serial_port.read()
        handle_serial_data(data)
    # write msg to UART serial port
    serial_port.write(close_hatch_cmd.encode())
    print('Hatch close command sent to MSP430')
    


# Function handles different data that is in the serial port buffer
# 1. Handle low feed levels msg -> push low feed notification -> send ack msg back
# 2. Handle non-low feed level msg -> send ack msg back (no further action required)
def handle_serial_data(data):
    if data == 'l'.encode():
        print("Feed is low! Sending notification")
        # send push notification for low bird feed warning
        title = 'Your birds are running out of food! ⚠️'
        message = "Your smart bird feeder is running low on bird feed.\nMake sure to refill it soon!"
        send_push_message(token, title, message)
        return

    if data == 'h'.encode():
        print("Feed is not low yet. No notification sent")
        return
    # TODO: Add all other serial port data checks below here (if any)


# Function that will write the current frame as a .jpg to local storage
def save_img(img, timestamp):
    cv2.imwrite("captured-bird-images/" + str('bird_memory' + ".jpeg"),
                cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))


def serial_config():
    serial_port = serial.Serial(
        port="/dev/ttyTHS1",
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )
    # Wait a second to let the port initialize
    time.sleep(1)

    print("Serial port is configured")

    return serial_port


# Function to update title bar of capture window
def update_title_bar(output, title):
    output.SetStatus(title)


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
    parser.add_argument("--threshold", type=float, default=0.75,
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

    # Set species to ignore to None to start off with so that no species is initially ignored.
    # This keeps track of what the last bird detected was.
    species_to_ignore = None

    print('Performing initial hatch open process')
    # initially, open the feed door, set hatch opened flag
    open_hatch(serial_port)
    hatch_is_open = True

    try:
        # capture initial time to track when the ultrasonic sensor should next be pulsed
        time1 = time.time()

        while True:
            time2 = time.time()

            data = ""

            # check if it is time to check feed levels
            if should_check_feed_lvl(time1, time2):
                # ask msp430 to read ultrasonic data and tell us if feed is low
                if serial_port.in_waiting > 0:
                    print('Data found in serial port in check #3') # debug
                    data = serial_port.read()
                    handle_serial_data(data)
                serial_port.write('u'.encode())
                print("'u' is sent #1")
                # reset waiting time for next pulse to ultrasonic
                time1 = time.time()

            # check if UART response indicates low feed
            # handle_serial_data(data)

            # TODO: ensure that this block executes
            if serial_port.in_waiting > 0:
                print('Data found in serial port in check #4') # debug
                data = serial_port.read()
                handle_serial_data(data)

            # check if MSP430 wants model to perform object detection
            # TODO: Indent everything after this if statement when IR detection trigger is added
#                if data == 'r'.encode():
#                    print('r received!')

            print('Starting loop for object detection... (Low feed asking & checking is still performed here).')

            # loop until serial port has stop message (received when MCU's sensor stops detecting presence)
            while True:
                # read serial port for stop message (received when MCU's sensor stops detecting objects)
                if serial_port.in_waiting > 0:
                    print('Data found in serial port in check #5') # debug
                    data = serial_port.read()
                    if data == 's'.encode():
                        print('Stop message received. Stopping object detection loop...')
                        # TODO: Add ack msg write here once Nikki gets it setup on MSP430
                        break
                    else:
                        handle_serial_data(data)
                else:
                    time2 = time.time()

                    # TODO: this is duplicate code. Figure out a way to refactor this.
                    # check if it is time to check feed levels
                    if should_check_feed_lvl(time1, time2):
                        if serial_port.in_waiting > 0:
                            print('Data found in serial port in check #6') # debug
                            data = serial_port.read()
                            handle_serial_data(data)
                        # ask msp430 to read ultrasonic data and tell us if feed is low
                        serial_port.write('u'.encode())
                        print("'u' is sent #2")
                        # reset waiting time for next pulse to ultrasonic
                        time1 = time.time()

                    species_to_ignore = run_obj_detection(
                        input, output, net, opt, serial_port, species_names, species_to_ignore)

    except KeyboardInterrupt:
        print("Exiting Program")

    except Exception as exception_error:
        print("Error occurred. Exiting Program")
        print("Error: " + str(exception_error))

    finally:
        serial_port.close()
        pass
