# USAGE
# python mask_rcnn_segmentation.py --input ../example_videos/dog_park.mp4 --output ../output_videos/mask_rcnn_dog_park.avi --display 0 --mask-rcnn mask-rcnn-coco
# python mask_rcnn_segmentation.py --input ../example_videos/dog_park.mp4 --output ../output_videos/mask_rcnn_dog_park.avi --display 0 --mask-rcnn mask-rcnn-coco --use-gpu 1

# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) input video file")
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
ap.add_argument("-m", "--mask-rcnn", required=True,
                help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-u", "--use-gpu", type=bool, default=0,
                help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
                               "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
                                "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
                               "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# check if we are going to use GPU
if args["use_gpu"]:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# initialize the video stream and pointer to output video file, then
# start the FPS timer
print("[INFO] accessing image stream...")

img = cv2.imread("testimage1.jpeg")

#print(img)
fps = FPS().start()

blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)

net.setInput(blob)
(boxes, masks) = net.forward(["detection_out_final",
                                  "detection_masks"])

# loop over the number of detected objects
for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the
        # confidence (i.e., probability) associated with the
        # prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the frame and then compute the width and the
            # height of the bounding box
            (H, W) = img.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object,
            # resize the mask such that it's the same dimensions of
            # the bounding box, and then finally threshold to create
            # a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                              interpolation=cv2.INTER_CUBIC)
            mask = (mask > args["threshold"])

            # extract the ROI of the image but *only* extracted the
            # masked region of the ROI
            roi = img[startY:endY, startX:endX][mask]

            # grab the color used to visualize this particular class,
            # then create a transparent overlay by blending the color
            # with the ROI
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original frame
            img[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the frame
            color = [int(c) for c in color]
            cv2.rectangle(img, (startX, startY), (endX, endY),
                          color, 2)

            # draw the predicted label and associated probability of
            # the instance segmentation on the frame
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(img, text, (startX, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
