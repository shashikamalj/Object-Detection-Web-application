"""
This app setups a client link to server app and provides video file streaming over this interface.
This stream is forwarded onto port 5555 of the server which is specified in the arguments [-s] with IP address
"""

# importing all the required packages
from imutils.video import VideoStream
import imutils
import imagezmq
import argparse
import socket
import time
import cv2

"""
Construct the argument parser and parse the arguments
"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())

"""
Initialize the ImageSender object with the socket address of the server
"""
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
	args["server_ip"]))

# Get the host name, initialize the video stream, and allow the camera sensor to warmup
rpiName = socket.gethostname()
print(rpiName)

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

"""
Initialize the ImageSender object with the socket address of the server
Send using imagezmq
"""
while(vs.isOpened()):
    # read the frame from the camera and send it to the server
    (grabbed, frame) = vs.read()
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    sender.send_image(rpiName, frame)
