import os
import logging
import extract_coods


image_file_op = open("output.txt", "w+")
dict_of_images = {}
dict_of_images_pred = {}

logging.basicConfig(filename='skipped_files.log', filemode='w', format='%(levelname)s - %(message)s')
increment = 10


def wrapper_for_image_manip(data_frame):
    """
    create a list of the bounding box coordinates based on the data frame receives. The wrapper converts the
    dataframe to a list to make it compatible with ImageManipulator functions
    :param data_frame: contains the bounding box coordinates
    :return: list_of_coods: list of bounding box coordinates
    """
    list_of_coods = []
    for i_row in range(len(data_frame)):
        x1 = data_frame.loc[i_row, "x1"]
        y1 = data_frame.loc[i_row, "y1"]
        x2 = data_frame.loc[i_row, "x2"]
        y2 = data_frame.loc[i_row, "y2"]
        list_of_coods.append([x1,y1,x2,y2])

    return list_of_coods


def wrapper_for_image_manip_pred(data_frame):
    """
    based on the data frame for the model identified objects, separate the coordinates of the boundary
    boxes and the confidence scores. This is needed for compatibility with ImageManipulator functions
    :param data_frame: Data frame consisting of the bounding boxes and the corresponding confidence scores
    for the detections of the model
    :return: boxes: list of bounding box coordinates
    :return: list_of_scores: list of confidence scores
    """
    boxes = wrapper_for_image_manip(data_frame)
    list_of_scores = []
    for i_row in range(len(data_frame)):
        confidence = data_frame.loc[i_row, "confidence"]
        list_of_scores.append(confidence)

    return boxes, list_of_scores


def run_fast_scandir(popup, directory, ext):  # dir: str, ext: list
    """
    we perform a recursive search through all the folders perform the model analysis for every image-file pair
    located. The results of this analysis is then converted to lists which are compatible with the
    ImageManipulator functions
    :param popup: popup gui
    :param directory: the path of the directory containing the image files
    :param ext: extensions allowed
    :return: None
    """
    subfolders, files = [], []

    total_no_of_detections = 0
    # go through all the contents of the folder
    for f in os.scandir(directory):
        # check if the given object is a directory
        if f.is_dir():
            # append it to the 'subfolders' list so that you can visit them later on
            subfolders.append(f.path)

        # check if the given object is a file with the requisite extension ['jpg','jpeg']
        if f.is_file() and os.path.splitext(f.name)[1].lower() in ext:

            # image_file_op: contains the names of all the file visited
            image_file_op.write(str(f.name)+"\n")
            # replace only the extension of an eligible file with txt
            if os.path.splitext(f.name)[1].lower() == ".jpg":
                txt_file_selected = f.path.replace(".jpg",".txt")
            else:
                txt_file_selected = f.path.replace(".jpeg", ".txt")
            # perform the model based detection on the file selected.
            model_coods = extract_coods.openncv_op(f.path)
            # reformat the contents in the 'txt_file_selected' to a dataframe, user_coods
            user_coods = extract_coods.user_defined_coordinate(txt_file_selected, f.path)

            # reformat the variables to make them compatible with the ImageManipulator functions
            user_coods = wrapper_for_image_manip(user_coods)
            model_coods_boxes, scores = wrapper_for_image_manip_pred(model_coods)

            # create a dictionary containing the user defined coordinates, bounding boxes identified by the
            # model an their confidence scores. Such a dictionary is necessary for the ImageManipulator functions
            dict_of_images[f.name] = user_coods
            dict_of_images_pred[f.name] ={"boxes":model_coods_boxes,"scores":scores}
            total_no_of_detections += len(scores)

    # visit all the subfolders detected
    for directory in list(subfolders):
        run_fast_scandir(popup, directory, ext)
    image_file_op.close()
    # print("total_no_of_detections",total_no_of_detections)
    return dict_of_images,dict_of_images_pred

