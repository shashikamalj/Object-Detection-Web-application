# USAGE
# Source: https://pysource.com/2020/04/02/train-yolo-to-detect-a-custom-object-online-with-free-gpu/
# python ssd_object_detection_img.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --input ../example_videos/testing_img1.jpg --output ../output_Images/output2.jpeg --display 0

# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
				help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
				help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str, default="",
				help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
				help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
				help="whether or not output frame should be displayed")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
				help="minimum probability to filter weak detections")
ap.add_argument("-u", "--use-gpu", type=bool, default=False,
				help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		   "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# check if we are going to use GPU
if args["use_gpu"]:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# initialize the video stream and pointer to output video file, then
# start the FPS timer
print("[INFO] accessing Image stream...")
img = cv2.imread("car.jpg")

# print(img)

fps = FPS().start()

# resize the frame, grab the frame dimensions, and convert it to
# a blob
img = imutils.resize(img, width=400)
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# extract the index of the class label from the
		# `detections`, then compute the (x, y)-coordinates of
		# the bounding box for the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# draw the prediction on the frame
		label = "{}: {:.2f}%".format(CLASSES[idx],
									 confidence * 100)
		cv2.rectangle(img, (startX, startY), (endX, endY),
					  COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(img, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

		cv2.imwrite("output.jpg", img)

# check to see if the output frame should be displayed to our
# screen
if args["display"] > 0:
	# show the output frame
	cv2.imshow("Frame", img)
	key = cv2.waitKey(1) & 0xFF

# update the FPS counter
fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))