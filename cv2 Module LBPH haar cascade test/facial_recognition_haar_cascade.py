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
# 1. Radius: the radius is used to build the circular local binary pattern and represents the
# radius around the central pixel. It is usually set to 1.
#
# 2. Neighbors: the number of sample points to build the circular local binary pattern.
# The more sample points you include, the higher the computational cost. Max value is 8.
#
# 3. Grid X: the number of cells in the horizontal direction. The more cells, the finer the grid,
# the higher the dimensionality of the resulting feature vector. Max value is 8
#
# 4. Grid Y: the number of cells in the vertical direction. The more cells, the finer the grid,
# the higher the dimensionality of the resulting feature vector. Max value is 8.
#
#############################################################################################
recognizer = cv2.face.LBPHFaceRecognizer_create(1, 4, 4, 4)


#############################################################################################
# A Haar Cascade is a machine learning object detection algorithm used to identify objects
# in an image or video and based on the concept of ​​ features.
#
# reference: https://ieeexplore.ieee.org/document/990517
#############################################################################################

# Load the Haar Cascade Classifier for frontal face images
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the Haar Cascade Classifier for face images with eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the Haar Cascade Classifier for face images with noses
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')

# Load the Haar Cascade Classifier for face images with mouths
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


def detect_single_face(image, faces):
    '''
    This function is designed to draw a bounding rectangle around the facial area of a single person
    contained in an image.

    :param image: image is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param faces: faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    :return: image with rectangle draw around facial area
    '''
    # (x,y) are the top-left coordinate of the rectangle
    # (w,h) are the width and height of the rectangle
    for (x, y, w, h) in faces:
        # Draw bounding rectangle based on parameter dimensions
        # BGR color values (3 parameters)
        # BGR color (255, 0, 255) - https://rgb.to/255,0,255
        # Line width in pixels
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    display_facial_detection_results(image)


def detect_multiple_faces(image, faces):
    '''
    This function is designed to draw a bounding rectangle around the facial area of multiple people
    contained in an image.

    :param image: image is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param faces: faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    :return: image with rectangle draw around facial areas
    '''
    # (x,y) are the top-left coordinate of the rectangle
    # (w,h) are the width and height of the rectangle
    for (x, y, w, h) in faces:
        # Draw bounding rectangle based on parameter dimensions
        # BGR (Blue, Green, Red) color values (3 parameters)
        # BGR color (255, 0, 255) - https://rgb.to/255,0,255
        # Line width in pixels
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    display_facial_detection_results(image)


def detect_eyes_single_face(image, faces, gray):
    '''
    This function is designed to draw a bounding rectangle around the eye area in the facial area
    of a single person contained in an image.

    :param image: image is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param faces: faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    :param gray: image color conversion, which is gray scale
    :return: image with rectangle draw around eye area
    '''
    # (x,y) are the top-left coordinate of the rectangle
    # (w,h) are the width and height of the rectangle
    for (x, y, w, h) in faces:

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = gray[y:y+h, x:x+w]

        # roi_color is a numpy.ndarray based on the color scale of the image
        roi_color = image[y:y+h, x:x+w]

        # eyes contains the calculate facial coordinates produced by eyes_cascade.detectMultiScale in relation to face_cascade
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)

        for (ex, ey, ew, eh) in eyes:
            # Draw bounding rectangle based on parameter dimensions around eyes
            # BGR (Blue, Green, Red) color values (3 parameters)
            # RGB color (255, 0, 128) - https://rgb.to/128,0,255
            # Line width in pixels
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (128, 0, 255), 2)

    display_facial_detection_results(image)


def detect_nose_single_face(image, faces, gray):
    '''
    This function is designed to draw a bounding rectangle around the nose area in the facial area
    of a single person contained in an image.

    :param image: image is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param faces: faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    :param gray: image color conversion, which is gray scale
    :return: image with rectangle draw around nose area
    '''
    # (x,y) are the top-left coordinate of the rectangle
    # (w,h) are the width and height of the rectangle
    for (x, y, w, h) in faces:

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = gray[y:y+h, x:x+w]

        # roi_color is a numpy.ndarray based on the color scale of the image
        roi_color = image[y:y+h, x:x+w]

        # nose contains the calculate facial coordinates produced by nose_cascade.detectMultiScale in relation to face_cascade
        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE)

        for (nx, ny, nw, nh) in nose:
            # Draw bounding rectangle based on parameter dimensions around nose
            # BGR (Blue, Green, Red) color values (3 parameters)
            # RGB color (255, 0, 0) - https://rgb.to/255,0,0
            # Line width in pixels
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)

    display_facial_detection_results(image)


def detect_mouth_single_face(image, faces, gray):
    '''
    This function is designed to draw a bounding rectangle around the mouth area in the facial area
    of a single person contained in an image.

    :param image: image is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param faces: faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    :param gray: image color conversion, which is gray scale
    :return: image with rectangle draw around moutharea
       '''
    # (x,y) are the top-left coordinate of the rectangle
    # (w,h) are the width and height of the rectangle
    for (x, y, w, h) in faces:

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = gray[y:y+h, x:x+w]

        # roi_color is a numpy.ndarray based on the color scale of the image
        roi_color = image[y:y+h, x:x+w]

        # mouth contains the calculate facial coordinates produced by mouth_cascade.detectMultiScale in relation to face_cascade
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)

        for (mx, my, mw, mh) in mouth:
            # Draw bounding rectangle based on parameter dimensions around mouth
            # BGR (Blue, Green, Red) color values (3 parameters)
            # RGB color (0, 255, 128) - https://rgb.to/0,255,128
            # Line width in pixels
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 128), 2)

    display_facial_detection_results(image)


def detect_single_face_multiple_features(image, faces, gray):
    '''
    This function is designed to draw a bounding rectangles around the face, eyes, nose and
    mouth areas in the facial area of a single person contained in an image.

    :param image: image is a numpy.ndarray, which is an array object represents a multidimensional,
                  homogeneous array of fixed-size items
    :param faces: faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    :param gray:  image color conversion, which is gray scale
    :return: image with rectangle draw around all facial areas
    '''
    # (x,y) are the top-left coordinate of the rectangle
    # (w,h) are the width and height of the rectangle
    for (x, y, w, h) in faces:
        # Draw bounding rectangle based on parameter dimensions
        # BGR (Blue, Green, Red) color values (3 parameters)
        # BGR color (255, 0, 255) - https://rgb.to/255,0,255
        # Line width in pixels
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)

        # roi_gray is a numpy.ndarray based on the gray scale of the image
        roi_gray = gray[y:y+h, x:x+w]

        # roi_color is a numpy.ndarray based on the color scale of the image
        roi_color = image[y:y+h, x:x+w]

        # eyes contains the calculate facial coordinates produced by eyes_cascade.detectMultiScale in relation to face_cascade
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.3, minNeighbors=4, flags = cv2.CASCADE_SCALE_IMAGE)

        # nose contains the calculate facial coordinates produced by nose_cascade.detectMultiScale in relation to face_cascade
        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor = 1.3, minNeighbors=8, flags = cv2.CASCADE_SCALE_IMAGE)

        # mouth contains the calculate facial coordinates produced by mouth_cascade.detectMultiScale in relation to face_cascade
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor = 1.3, minNeighbors=4, flags = cv2.CASCADE_SCALE_IMAGE)

        for (ex, ey, ew, eh) in eyes:
            # Draw bounding rectangle based on parameter dimensions
            # BGR (Blue, Green, Red) color values (3 parameters)
            # BGR color (128, 0, 255) - https://rgb.to/128,0,255
            # Line width in pixels
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (128, 0, 255), 2)

        for (nx, ny, nw, nh) in nose:
            # Draw bounding rectangle based on parameter dimensions around nose
            # BGR (Blue, Green, Red) color values (3 parameters)
            # BGR color (255, 0, 0) - https://rgb.to/255,0,0
            # Line width in pixels
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)

        for (mx, my, mw, mh) in mouth:
            # Draw bounding rectangle based on parameter dimensions
            # BGR (Blue, Green, Red) color values (3 parameters)
            # BGR color (0, 255, 128) - https://rgb.to/0,255,128
            # Line width in pixels
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 128), 2)

    display_facial_detection_results(image)


def display_facial_detection_results(image):
    # Display image with bounding rectangles
    # and title in a window. The window
    # automatically fits to the image size.
    cv2.imshow('Facial feature recognition', image)
    # Displays the window infinitely
    key = cv2.waitKey(0)
    # Shuts down the display window and terminates
    # the Python process when the key 'q' is pressed
    if key == ord('q'):
        cv2.destroyAllWindows()


def process_single_image(name):
    '''
    This function is designed process a single image and calculate the facial coordinates produced
    by face_cascade.detectMultiScale

    :param name: the name of the image file to use
    :return: image is a numpy.ndarray, which is an array object represents a multidimensional,
             homogeneous array of fixed-size items

             gray is the image color conversion, which is gray scale

             faces contains the calculate facial coordinates produced by face_cascade.detectMultiScale
    '''
    # Read in the image
    image = cv2.imread(name)
    if image is not None:

        # cv2.cvtColor(input_image, flag) where flag determines the type of conversion.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detectMultiScale detects objects of different sizes in the input image.
        # The detected objects are returned as a list of rectangles.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        return image, gray, faces

    else:
        print(f'The following image could not be read by OpenCV: {name}')


def validate_file_extension(filename):
    accepted_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tiff')
    if filename.endswith(accepted_extensions):
        return True
    else:
        return False


image_name = 'natalie_portman.jpeg'
valid_file_extension = validate_file_extension(image_name)

if valid_file_extension == True:
    face_results = process_single_image(image_name)
    if face_results:
        image = face_results[0]
        gray = face_results[1]
        faces = face_results[2]
        detect_single_face(image, faces)

elif valid_file_extension == False:
    print(f'The following image does not have a valid file_extension {image_name}')








