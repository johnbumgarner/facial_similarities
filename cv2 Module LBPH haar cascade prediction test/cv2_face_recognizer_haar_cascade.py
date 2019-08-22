#############################################################################################
# The OS module in provides functions for interacting with the operating system.
#
# OS.walk() generate the file names in a directory tree by walking the tree.
#############################################################################################
import os
from os import walk

#############################################################################################
# The pickle module is used for serializing and de-serializing a Python object structure.
# Pickling is a way to convert a python object (list, dict, etc.) into a character stream.
# The data format used by pickle is Python-specific.
#############################################################################################
import pickle

#############################################################################################
# numpy is one of fundamental packages for scientific computing with Python
#############################################################################################
import numpy as np

#############################################################################################
# The OpenCV is a library of programming functions mainly aimed at real-time computer vision.
# The Python package is opencv-python
#
# reference: https://pypi.org/project/opencv-python
#############################################################################################
import cv2

#############################################################################################
# LBPH face recognition algorithm
#
# Local Binary Pattern (LBP) is a simple yet very efficient texture operator
# which labels the pixels of an image by thresholding the neighborhood of each
# pixel and considers the result as a binary number.
#
# Parameters: the LBPH uses 4 parameters:
#
# 1. Radius: the radius used for building the Circular Local Binary Pattern. The greater the
# radius, the smoother the image but more spatial information you can get.
#
# 2. Neighbors: the number of sample points to build a Circular Local Binary Pattern from.
# An appropriate value is to use 8 sample points. Keep in mind: the more sample points
# you include, the higher the computational cost. Max value is 8.
#
# 3. Grid X: the number of cells in the horizontal direction, 8 is a common value used in publications.
# The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
# Max value is 8
#
# 4. Grid Y: 	The number of cells in the vertical direction, 8 is a common value used in publications.
# The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
# Max value is 8.
#
#############################################################################################
recognizer = cv2.face.LBPHFaceRecognizer_create(radius = 1, neighbors = 4,  grid_x = 4,  grid_y = 4)

#############################################################################################
# A Haar Cascade is a machine learning object detection algorithm used to identify objects
# in an image or video and based on the concept of ​​ features.
#
# reference: https://ieeexplore.ieee.org/document/990517
#############################################################################################

# Load the Haar Cascade Classifier for frontal face images
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_image_files(directory_of_images):
    '''
     This function is designed to traverse a directory tree and extract all
     the image names contained in the directory.

    :param directory_of_images: the name of the target directory containing
           the images to be trained on.
    :return: list of images to be processed.
    '''
    images_to_process = []
    for (dirpath, dirnames, filenames) in walk(directory_of_images):
        for filename in filenames:
            accepted_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tiff')
            if filename.endswith(accepted_extensions):
                images_to_process.append(os.path.join(dirpath, filename))
        return images_to_process


def prepare_training_data(images, image_height, image_width):
    '''
     This function is designed to generate the data arrays used for labelling
     and the FaceRecognizer process.

     ref: https://docs.opencv.org/3.4/dd/d65/classcv_1_1face_1_1FaceRecognizer.html#a3182081e5f8023e658ad8ab96656dd63

    :param images: the names of the images that will be used in the training data
    :param image_height: this height attribute specifies the height of an image, in pixels
    :param image_width: this width attribute specifies the width of an image, in pixels
    :return: the image_names
             x_train contains the training data
             y_labels contains the association between the images and their labels.
    '''

    # ID numbers associated with image labels
    current_id = 0

    # unique set of image names
    image_names = {}

    # list used to store ID numbers and image label information
    y_labels = []

    # list used to store the training data
    x_train = []

    for image in images:
        name = os.path.split(image)[-1].split(".")[0]
        if not name in image_names:
            image_names[name] = current_id
            current_id += 1
            id_ = image_names[name]

            # Read in the image
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

            # resizes the image to preestablished height and width dimensions
            img = cv2.resize(img, (image_height, image_width), interpolation=cv2.INTER_AREA)

            # cv2.cvtColor(input_image, flag) where flag determines the type of conversion
            grayscale_image = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

            # faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
            faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor = 1.3, minNeighbors = 8, flags = cv2.CASCADE_SCALE_IMAGE)

            # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
            # (width, height) are the width and height of the rectangle
            for (x_coordinate, y_coordinate, width, height) in faces:
                print (f'Processing training data for image: {name}')

                # roi_gray is a numpy.ndarray based on the gray scale of the image
                roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

                # append the training data to the x_train list
                x_train.append(roi_gray)

                # append the label information to the y_labels list
                y_labels.append(id_)

    return image_names, x_train, y_labels

def create_training_data(image_names, x_train, y_labels):
    '''
     This function is designed to generate the files used for labelling
     and the FaceRecognizer process.

    :param image_names: the names of the images that are in the training data
    :param x_train: the numpy.ndarray related to the images used in the training process.
    :param y_labels: labels associated with the images
    :return: pickle file containing the association between the images and their labels.
             YML file used by the FaceRecognizer process.
    '''
    with open('face_labels.pickle', 'wb') as pickle_file:
        print ('Creating relational information for the training data.')
        pickle.dump(image_names, pickle_file)

    # Trains the FaceRecognizer with given data and associated labels.
    # The training images, that means the faces you want to learn.
    # The image data and their corresponding labels have to be given
    # as a vector.
    recognizer.train(x_train, np.array(y_labels))
    print ('Writing out the image training data.')

    # Saves a FaceRecognizer and its model state to a
    # given filename, either as XML or YML (aka YAML)
    recognizer.write('face_train_data.yml')
    print('Finished processing the image training data.')


def facial_recognition(image_name, image_height, image_width):
    '''
    This function is designed to perform facial prediction on a target
    images against the training data previously generated.

    :param image_name: the name of the image that will be processed
    :param image_height: this height attribute specifies the height of an image, in pixels
    :param image_width: this width attribute specifies the width of an image, in pixels
    :return: image with rectangle draw around facial area overlaid with image label and
             confidence score
    '''

    # Loads a persisted FaceRecognizer model and state from a YML (aka YAML) file
    recognizer.read('face_train_data.yml')

    # Sets the font parameters to use with cv2.putText
    font = cv2.FONT_HERSHEY_SIMPLEX
    fntSize = 0.60
    fntThickness = 2
    fntColor = (0, 255, 0)

    # Open the pickle file containing the image names
    # and their associated labels,
    with open('face_labels.pickle', 'rb') as pickle_file:
        # loads the pickle file
        pickle_labels = pickle.load(pickle_file)
        # extract the key and values pairs from the pickle file.
        labels = {value:key for key,value in pickle_labels.items()}

    # Read in the image
    image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)

    # resizes the image to preestablished height and width dimensions
    image = cv2.resize(image, (image_height, image_width), interpolation=cv2.INTER_AREA)

    # cv2.cvtColor(input_image, flag) where flag determines the type of conversion.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor = 1.3, minNeighbors = 8, flags = cv2.CASCADE_SCALE_IMAGE)

    # (x_coordinate, y_coordinate) are the top-left coordinate of the rectangle
    # (width, height) are the width and height of the rectangle
    for (x_coordinate, y_coordinate, width, height) in faces:

        # Draw bounding rectangle based on parameter dimensions
        # BGR color values (3 parameters)
        # BGR color (0, 255, 0) - https://rgb.to/0,255,0
        # Line width in pixels
        cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + width, y_coordinate + height), (255, 0, 255), 2)

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = grayscale_image[y_coordinate:y_coordinate + height, x_coordinate:x_coordinate + width]

        # predicts a label and associated confidence (e.g. distance) for a given input image.
        id, confidence_face = recognizer.predict(roi_gray)

        # The confidence score is adjustable, but setting the score
        # to high can produce inviald results.
        if confidence_face <= 10:

            # extract the id number associated with a specific image label
            name = labels[id]

            # create the text to be overlaid on the image
            label_info = (f'Name: {name} \n\nConfidence score: {confidence_face}')

            # sets the starting location of the text
            y0, dy = 15, 12

            # splits the label information based on new lines
            for i, line in enumerate(label_info.split('\n')):

                # caluates the y_coordinate of the text
                y_coordinate = y0 + i * dy

                # the function putText renders the specified text string in the image.
                cv2.putText(image, line, (10, y_coordinate), font, fntSize, fntColor, fntThickness)

        elif confidence_face > 10:
            name = 'Unknown Person'

            # the function putText renders the specified text string in the image.
            cv2.putText(image, name, (x_coordinate - 24, y_coordinate - 24), font, fntSize, fntColor, fntThickness, cv2.LINE_AA)

    display_facial_prediction_results(image)


def display_facial_prediction_results(image):
    # Display image with bounding rectangles
    # and title in a window. The window
    # automatically fits to the image size.
    cv2.imshow('Facial Prediction', image)
    # Displays the window infinitely
    key = cv2.waitKey(0) & 0xff
    # Shuts down the display window and terminates
    # the Python process when a key is pressed on
    # the window.
    if key == 113 or key == 27: # press 'ESC' to quit
        cv2.destroyAllWindows()


image_directory = 'front_facing_images'

images = get_image_files(image_directory)
image_height = 300
image_width = 300

# assemble the training data arrays and the associated labels
training_data = prepare_training_data(images, image_height, image_width)
if training_data:
    images = training_data[0]
    training =  training_data[1]
    label_info = training_data[2]

    # generate the training data and the associated labels
    create_training_data(images, training, label_info)

    ############################################################
    # This code below is running a single face prediction test.
    ###########################################################
    image_name = 'aishwarya_rai.jpg'
    facial_recognition(image_name, image_height, image_width)