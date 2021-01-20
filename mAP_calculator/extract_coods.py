"""
The app returns a dataframe consisting of the bounding box coordinates of the detections and the confidence values
based on the object detection model chosen. We have analysed - YOLO, RCNN, SSD
The program utilises elements from:
https://www.pyimagesearch.com/2020/02/10/opencv-dnn-with-nvidia-gpus-1549-faster-yolo-ssd-and-mask-r-cnn/
The functions for the different detection models were obtained from here
"""
import pandas
import os
import logging
import cv2
import numpy as np
import imutils
from imutils.video import FPS


### basic setup ###
logging.basicConfig(filename='skipped_files.log', filemode='w', format='%(levelname)s - %(message)s')
increment = 10

net = cv2.dnn.readNet("res/yolov3.weights", "res/yolov3.cfg")
#####################


def openncv_op(file):
    """
    common interface for choosing the proper model to perform the detection
    :param file: contains the image file to perform the detection on
    :return: model_defined_cood: a dataframe consisting of bounding box coordinates and
    corresponding confidence values
    """
    # change the function name depending on the model chosen
    model_defined_cood = ssd_based_detection(file)
    return model_defined_cood


def user_defined_coordinate(txt_file_selected, img_file_selected):
    """
    The program creates a dataframe based on the bounding box values obtained in the 'txt_file_selected' for the
    corresponding 'img_file_selected'

    :param txt_file_selected: the text file containing the bounding box coordinates
    :param img_file_selected: the image file based on which the 'txt_file_selected' was created
    :return: user_defined_cood: a dataframe consisting of bounding box coordinates and
    corresponding confidence values
    """
    list_coordinates_new = [[], [], [], [], []]
    with open(txt_file_selected, "r") as cood_file:
        list_coordinates = cood_file.readlines()

    # convert each of the coordinate values to float
    for each_cood in list_coordinates:
        list_coods = each_cood.split()

        for i in range(0, len(list_coordinates_new)):
            list_coordinates_new[i].append(float(list_coods[i]))

    # denotes the column headers for ths contents of txt_file_selected
    # lbl = label, x = x center of boundary box, y = y center of boundary box,
    # w = width, h = height
    list_headers = ['lbl', 'x', 'y', 'w', 'h']

    # cood_data_frame to store the contents of the txt_file_selected as a dataframe
    cood_data_frame = pandas.DataFrame(columns=list_headers)
    i_length = 0
    for each_header in list_headers:
        cood_data_frame[each_header] = list_coordinates_new[i_length]
        i_length += 1

    # sort the cood_data_frame values based on x,y
    cood_data_frame = cood_data_frame.sort_values(['x', 'y'], ascending=[True, True])

    img_file = cv2.imread(img_file_selected, cv2.IMREAD_UNCHANGED)
    img_file = cv2.resize(img_file, None, fx=0.2, fy=0.2)
    image = img_file
    # Blue color in BGR
    color = (255, 0, 0)
    user_defined_cood = pandas.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])

    # Line thickness of 2 px
    thickness = 2
    for i in range(len(cood_data_frame)):
        # start_x = x-center - w/2, start_y = y-center - h/2
        start_x = int((cood_data_frame.loc[i, "x"] - (cood_data_frame.loc[i, "w"]) / 2) * img_file.shape[1])
        start_y = int((cood_data_frame.loc[i, "y"] - cood_data_frame.loc[i, "h"] / 2) * img_file.shape[0])

        # end_x = start_x + width, end_y = start_y + height
        end_x = start_x + (int(cood_data_frame.loc[i, "w"] * img_file.shape[1]))
        end_y = start_y + (int(cood_data_frame.loc[i, "h"] * img_file.shape[0]))
        start_point = (start_x, start_y)
        values_to_add = {'x1': start_x, 'y1': start_y, "x2": end_x, "y2": end_y}
        user_defined_cood = user_defined_cood.append(values_to_add, ignore_index=True)
        # uncomment to view images with the boundary boxes and save detection in output_imgs/ folder
        """
        end_point = (end_x, end_y)

        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        # print(user_defined_cood)
    window_name = 'Image'
    # Displaying the image
    # cv2.imshow(window_name, image)
    # cv2.waitKey(0)
    print("file_name", img_file_selected.split("/")[-1].split("\\")[-1])
    cv2.imwrite("output_imgs_user/" + str(img_file_selected.split("/")[-1].split("\\")[-1]), image)
    # cv2.destroyAllWindows()
    """
    user_defined_cood = user_defined_cood.sort_values(['x1', 'y1', 'x2', 'y2'], ascending=[True, True, True, True])
    return user_defined_cood


def yolo_based_detection(file):
    """
    Perform object detection using YOLO
    :param file: contains the image file to perform the detection on
    :return: model_defined_cood: a dataframe consisting of bounding box coordinates and corresponding confidence values
    """

    classes = []
    with open("res/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # get the output layers, the final result, to be displayed on screen

    # each channel goes from 0-255 - put random color in each channel,
    # generate as many colors as classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # create empty dataframe to store the bounding box coordinates along with the confidence values
    model_defined_cood = pandas.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'confidence'])
    img = cv2.imread(file)
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    #fps = FPS().start()
    # any object detection with a confidence score of atleast 70% is considered
    confidence_limit = 0.7

    height, width, channels = img.shape

    # True changes channel from BGR to RGB
    # we have a blob for each of the channels - red, green and blue
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)

    net.setInput(blob)
    # we want to forward it to the output layer to get the result
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # consider those detections with confidence > confidence_limit
            if confidence > confidence_limit:
                # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle Coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                values_to_add = {'x1': x, 'y1': y, "x2": x + w, "y2": y + h, "confidence": confidence}
                model_defined_cood = model_defined_cood.append(values_to_add, ignore_index=True)

                # uncomment to view images with the boundary boxes and save detection in output_imgs/ folder
                """
                confidences.append(float(confidence))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                class_ids.append(class_id)
                
    font = cv2.QT_FONT_NORMAL
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        if i in indexes:
            label = str(classes[class_ids[i]])
            color = colors[i]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 3)
            #print(label)
    print(indexes)
    cv2.imshow("img", img)
    cv2.waitKey()
    print("file_name",file.split("/")[-1].split("\\")[-1])
    cv2.imwrite("output_imgs/"+str(file.split("/")[-1].split("\\")[-1]),img)
    """
    # sort model_defined_cood based on coordinate values
    # fps.update()
    # fps.stop()
    # print("{:.2f}".format(fps.fps()))
    model_defined_cood = model_defined_cood.sort_values(['x1', 'y1', 'x2', 'y2'], ascending=[True, True, True, True])

    return model_defined_cood


def ssd_based_detection(file):
    """
    Perform object detection using SSD
    :param file: contains the image file to perform the detection on
    :return: model_defined_cood: a dataframe consisting of bounding box coordinates and corresponding confidence values
    """
    net = cv2.dnn.readNetFromCaffe("./res/ssd/MobileNetSSD_deploy.prototxt", "./res/ssd/MobileNetSSD_deploy.caffemodel")
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    model_defined_cood = pandas.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'confidence'])
    img = cv2.imread(file)
    img = img = imutils.resize(img, width=813)
    # the confidence limit was needed to be this low to get any type of detection. This may probably be since
    # we are using MobileNet_SSD model
    confidence_limit = 0.3

    h, w, channels = img.shape
    #fps = FPS().start()
    # True changes channel from BGR to RGB
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)

    net.setInput(blob)
    # we want to forward it to the output layer to get the result
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_limit:
            # extract the index of the class label from the
            # 'detections', then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            values_to_add = {'x1': startX, 'y1': startY, "x2": endX, "y2": endY, "confidence": confidence}
            model_defined_cood = model_defined_cood.append(values_to_add, ignore_index=True)

            # uncomment to view images with the boundary boxes and save detection in output_imgs/ folder
            """
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            cv2.rectangle(img, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Frame", img)
    key = cv2.waitKey(0)
    print("file_name",file.split("/")[-1].split("\\")[-1])
    cv2.imwrite("output_imgs/"+str(file.split("/")[-1].split("\\")[-1]),img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    """
    #fps.update()
    #fps.stop()
    #print("{:.2f}".format(fps.fps()))
    model_defined_cood = model_defined_cood.sort_values(['x1', 'y1', 'x2', 'y2'], ascending=[True, True, True, True])

    return model_defined_cood


def rcnn_based_detection(file):
    """
     Perform object detection using RCNN
    :param file:  contains the image file to perform the detection on
    :return: model_defined_cood: a dataframe consisting of bounding box coordinates and corresponding confidence values
    """
    # load the COCO class labels our Mask R-CNN was trained on
    labelsPath = os.path.sep.join(["./res/mask-rcnn-coco",
                                   "object_detection_classes_coco.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath = os.path.sep.join(["./res/mask-rcnn-coco",
                                    "frozen_inference_graph.pb"])
    configPath = os.path.sep.join(["./res/mask-rcnn-coco",
                                   "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    model_defined_cood = pandas.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'confidence'])
    img = cv2.imread(file)
    img = cv2.resize(img, None, fx=0.2, fy=0.2)

    confidence_limit = 0.7
    #fps = FPS().start()
    # True changes channel from BGR to RGB
    # blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)
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
        if confidence > confidence_limit:
            # scale the bounding box coordinates back relative to the
            # size of the frame and then compute the width and the
            # height of the bounding box
            (H, W) = img.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            values_to_add = {'x1': startX, 'y1': startY, "x2": endX, "y2": endY, "confidence": confidence}
            model_defined_cood = model_defined_cood.append(values_to_add, ignore_index=True)

            # uncomment to view images with the boundary boxes and save detection in output_imgs/ folder
            """
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object,
            # resize the mask such that it's the same dimensions of
            # the bounding box, and then finally threshold to create
            # a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                              interpolation=cv2.INTER_CUBIC)
            mask = (mask > 0.3)

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
            #cv2.imshow("img", img)
            #key = cv2.waitKey(0)

            # cv2.imwrite("output.jpg", img)

    # check to see if the output frame should be displayed to our
    # screen

    print("file_name",file.split("/")[-1].split("\\")[-1])
    cv2.imwrite("output_imgs/"+str(file.split("/")[-1].split("\\")[-1]),img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    """
    #fps.update()
    #fps.stop()
    #print("{:.2f}".format(fps.fps()))
    model_defined_cood = model_defined_cood.sort_values(['x1', 'y1', 'x2', 'y2'], ascending=[True, True, True, True])

    return model_defined_cood
