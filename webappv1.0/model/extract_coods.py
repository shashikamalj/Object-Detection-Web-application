"""
The app returns a dataframe consisting of the bounding box coordinates of the detections, dictionary
with vehicle frequencies and image frame containing the detections all done based on the YOLO detection model
"""
import pandas
from imutils.video import FPS
import logging
import cv2
import numpy as np

### basic setup ###
old_fps = 0.0
num = 1
logging.basicConfig(filename='skipped_files.log', filemode='w', format='%(levelname)s - %(message)s')
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
    model_defined_cood = yolo_based_detection(file)
    return model_defined_cood


def yolo_based_detection(frame):
    """
    Perform object detection using YOLO
    :param frame: contains the image file to perform the detection on
    :return: model_defined_cood: a dataframe consisting of bounding box coordinates and corresponding confidence values
    :return: dict_of_classes: a dictionary consisting of key values as the vehicle types detected
                            and the value denoting the frequency of the corresponding vehicle type in the frame
    :return: img: the image frame with the boundary boxes, class labels and measured FPS

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
    img = frame
    # fps = FPS().start()
    # any object detection with a confidence score of atleast 70% is considered
    confidence_limit = 0.7

    height, width, channels = img.shape

    # True changes channel from BGR to RGB
    # we have a blob for each of the channels - red, green and blue
    fps = FPS().start()
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)

    net.setInput(blob)
    # we want to forward it to the output layer to get the result
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    list_of_classes = []
    dict_of_classes = {}
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

                # store the coordinates of the boundary boxes
                boxes.append([x, y, w, h])
                # create a dictionary of boundary box coordinates
                values_to_add = {'x1': x, 'y1': y, "x2": x + w, "y2": y + h, "confidence": confidence}
                model_defined_cood = model_defined_cood.append(values_to_add, ignore_index=True)

                confidences.append(float(confidence))

                # collect the list of class labels
                list_of_classes.append(str(classes[class_id]))

                class_ids.append(class_id)

    fps.update()
    fps.stop()
    no_obj_detected = len(boxes)
    font = cv2.QT_FONT_NORMAL
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # iterate through all the boundary boxes identified
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        if i in indexes:
            label = str(classes[class_ids[i]])
            color = colors[i]

            # draw rectangles identifying boundary box, along with the label for the object
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 3)

    global old_fps, num

    # calculate FPS
    old_fps = (fps.fps()+old_fps)
    value = old_fps/num
    num+=1
    label ="Avg FPS:" + str("{:.2f}".format(value))

    cv2.putText(img, label, (25, 25), font, 1, (0,0,255), 2)

    # create a dictionary indicating the object class as the key and the frequency count
    # of the objects in the frame as the value
    for each_class in list_of_classes:
        dict_of_classes[each_class] = dict_of_classes.get(each_class, 0) + 1

    # sort model_defined_cood based on coordinate values
    model_defined_cood = model_defined_cood.sort_values(['x1', 'y1', 'x2', 'y2'], ascending=[True, True, True, True])

    return model_defined_cood,dict_of_classes,img

