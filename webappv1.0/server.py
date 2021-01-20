"""
This app setups a Flask server app and provides server streaming interface. These local clients streams are analysed using
object detection models and the output is streamed onto an interface where all the object detection processed streams
are visible to user
"""

# importing all the required packages
import numpy as np
import imagezmq
from flask import Flask, render_template, request
from flask import Response
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from model.extract_coods import openncv_op
import json
import imutils
import threading
import argparse
import cv2

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './image_upload/images/'
dict_of_classes_for_instance = {}
# Initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
lastActive = {}
lastActiveCheck = datetime.now()
frameDict = {}
# Stores the estimated number of Pis, active checking period, and
# calculates the duration seconds to wait before making a check to
# see if a device was active
ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

# Initialize the ImageHub object
imageHub = imagezmq.ImageHub()
def detect_frames():
    """
    Detecting frames from the client video stream
    :param none:
    :return: camera_name: a dataframe consisting of the client camera name
    """
    while True:
        # receive RPi name and frame from the RPi and acknowledge
        # the receipt
        global outputFrame, lock
        
        (camera_name, frame) = imageHub.recv_image()
        timestamp = datetime.now()
        timestamp = timestamp.strftime("%d-%b-%Y %H:%M:%S")
        imageHub.send_reply(b'OK')
        #frame = imutils.resize(frame, width=400)
        if camera_name not in lastActive.keys():
            print("[INFO] receiving data from {}...".format(camera_name))
		# record the last active time for the device from which we just
	    # received a frame
        lastActive[camera_name] = datetime.now()
               
        with lock:    
            model_defined_cood, dict_of_classes,outputFrame = openncv_op(frame)
            frameDict[camera_name] = outputFrame
            # build a montage using images in the frame dictionary
            outputFrame = cv2.vconcat(list(frameDict.values()))
            dict_of_classes['Camera Name'] = camera_name
            dict_of_classes_for_instance[timestamp] = dict_of_classes

#             for (i, montage) in enumerate(montages):
#                 outputFrame = np.concatenate((montage))
        print(camera_name)
        #print(model_defined_cood.to_json())
#         with lock:
#             outputFrame = frame.copy()

def generate():
    """
    Rendering video frames until end of video
    :param none:
    :return: yield: Render video frames in byte format
    """
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    """
    Index function, returns the rendered HTML landing page
    :param none:
    :return: render_template: Rendered HTML page
    """
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """
    Returning the response generated along with the specific media type(mime type)
    :param none:
    :return: Response: Video feed response with media type
    """
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/return_vehicle_frequencies', methods=['GET', 'POST'])
def return_file():
    """
    The program creates a JSON dataframe based on the detected bounding box probability values, classes and timestamp

    :param return_vehicle_frequencies: the URL argument passed to specify this funtion
    :param methods: the method POST and GET
    :return: return_dict_of_classes_for_instance: a json structure consisting of bounding box coordinates and
    corresponding confidence values
    """
    global dict_of_classes_for_instance
    with lock:
        return_dict_of_classes_for_instance = dict_of_classes_for_instance.copy()
        dict_of_classes_for_instance = {}
        return json.dumps(return_dict_of_classes_for_instance)

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_frames)
    t.daemon = True
    t.start()
    # Start the flask app with an open IP, port and threading for improved performace
    app.run(host='0.0.0.0', port=5860, debug=True, threaded=True, use_reloader=False)
