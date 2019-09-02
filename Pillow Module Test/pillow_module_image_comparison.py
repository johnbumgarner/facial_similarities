#############################################################################################
# The OS module in provides functions for interacting with the operating system.
#
# OS.walk() generate the file names in a directory tree by walking the tree.
#############################################################################################
import os
from os import walk

######################################################################################
# The Python module Pillow is the folk of PIL, the Python Imaging Library
# reference: https://pillow.readthedocs.io/en/3.0.x/index.html
######################################################################################
# This module is used to load images
from PIL import Image
# This module contains a number of arithmetical image operations
from PIL import ImageChops

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


def image_pixel_differences(base_image, compare_image):
    """
    Calculates the bounding box of the non-zero regions in the
    image.
    :param base_image: target image to find
    :param compare_image:  set of images containing the target image
    :return: The bounding box is returned as a 4-tuple defining the
             left, upper, right, and lower pixel coordinate. If the image
             is completely empty, this method returns None.
    """
    # Returns the absolute value of the pixel-by-pixel
    # difference between two images.
    diff = ImageChops.difference(base_image, compare_image)
    if diff.getbbox():
        return False
    else:
        return True


image_directory = 'female_headshots_with_earrings'
target_image_name = 'jennifer_aniston.jpeg'
target_image = Image.open(f'{image_directory}/{target_image_name}')

images = get_image_files(image_directory)

#####################################################
# Compares all the images that are contained in the
# image_directory against the target image.
#####################################################
for image in images:
    compare_image = Image.open(image)
    results = image_pixel_differences(target_image, compare_image)
    if results == True:
        print(f'These images have identical pixels: {target_image_name} -- {image}')
    elif results == False:
        print (f'These images have dissimilar pixels: {target_image_name} -- {image}')