"""
This is a GUI app for selecting a folder with images to quantify model performance via the
mean average precision(mAP) metric.
Please note that the folder should have the following structure:
There should be an image and text file pair with the same name. The text file should consist of
boundary box coordinates with label identifying the type of object in the boundary box (rectangular shape)
20200518_152832.jpg, 20200518_152832.txt
Content of 20200518_152832.txt:
0 0.400500 0.827000 0.157000 0.260667
0 0.536875 0.423333 0.059250 0.092667
0 0.441375 0.353667 0.040750 0.050000
.....

First column indicates the type of object. Successive columns indicate the boundary box coordinates
2nd and 3rd column indicate the x center and y center, width and height
"""
from tkinter.ttk import *
from tkinter import *
from tkinter import filedialog
import perform_operations

# importing the required module
import matplotlib.pyplot as plt
from image_manip import ImageManipulator

master = Tk()
master.title('File/Batch Extractor')
master.geometry("300x120")
var = IntVar()
var.set(1)
# extensions
ext = [".jpeg", ".jpg"]

var_choice = IntVar()
var_choice.set(1)


def plot_precision_recall(precision, recall):
    """
    Plot precision vs recall
    :param precision: list of precision values for the model
    :param recall: list of recall values for the model
    :return:
    """
    plt.plot(recall, precision)

    # naming the x axis
    plt.xlabel('recall - axis')
    # naming the y axis
    plt.ylabel('precision - axis')

    # giving a title to my graph
    plt.title('precision vs recall')

    # function to show the plot
    plt.show()


def perform_directory_extraction():
    """
    Choose directory to read image files
    :return: None
    """
    folder_selected = filedialog.askdirectory()

    # create popup
    popup = Toplevel()
    Label(popup, text="Please wait. Extraction under progress", font=(None, font_size)).grid(row=0, column=0)

    popup.pack_slaves()
    popup.geometry('280x50')

    popup.update()

    # recursively go through the selected folder and find the different image files
    dict_val_user, dict_val_model = perform_operations.run_fast_scandir(popup, folder_selected, ext)

    print("done")
    popup.destroy()

    # ImageManipulator object
    img = ImageManipulator()

    # perform the mAP operation to get the model performance
    dict_val = img.get_avg_precision_at_iou(dict_val_user, dict_val_model)
    print(dict_val)

    # plot a precision vs recall
    plot_precision_recall(dict_val['precisions'], dict_val['recalls'])


def perform_file_extraction():
    file_selected = filedialog.askopenfilename(title="Select file for extraction",
                                               filetypes=(("RPC File", (".tim", ".rsp")),))
    perform_operations.run_file_extractor(file_selected)


def perform_extraction(popup):
    # close the file choosing popup
    popup.destroy()

    selection = var.get()

    if selection == 1:
        perform_directory_extraction()
    else:
        perform_file_extraction()

    master.quit()


if __name__ == "__main__":
    font_size = 11
    # Option to 'select a directory'
    Label(master, text="Select type of TIM extraction", font=(None, font_size)).grid(row=0, sticky=W, columnspan=3)
    Radiobutton(master, text="Select a directory", variable=var, value=1, font=(None, font_size)).grid(row=1, sticky=W,
                                                                                                       columnspan=3)
    # Option to 'Select a file'
    # Radiobutton(master, text="Select a file", variable=var, value=2, font=(None, font_size)).grid(row=2, sticky=W,
    #                                                                                               columnspan=3)
    # on click, allows navigation to the file choose_save_file_formats fn.
    Button(master, text="OK", width=7, command=perform_directory_extraction, font=(None, font_size)).grid(row=5,
                                                                                                          sticky=W,
                                                                                                          column=2,
                                                                                                          columnspan=2)
    master.mainloop()
