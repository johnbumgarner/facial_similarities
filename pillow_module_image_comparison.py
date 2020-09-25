#!/usr/local/bin/python3

##################################################################################
# “AS-IS” Clause
#
# Except as represented in this agreement, all work produced by Developer is
# provided “AS IS”. Other than as provided in this agreement, Developer makes no
# other warranties, express or implied, and hereby disclaims all implied warranties,
# including any warranty of merchantability and warranty of fitness for a particular
# purpose.
##################################################################################

##################################################################################
#
# Date Completed: July 24, 2019
# Author: John Bumgarner
#
# Date Revised: September 24, 2020
# Revised by: John Bumgarner
#
# This Python script is designed to use the Pillow module to calculate the pixel-by-pixel
# difference between two images. If the images have similar pixel values then the
# images are identical, but if the values are different then they are dissimilar.
#
##################################################################################

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
    """
    This function is designed to traverse a directory tree and extract all
    the image names contained in the directory.

    :param directory_of_images: the name of the target directory containing
           the images to be trained on.
    :return: list of images to be processed.
    """
    images_to_process = []
    for (dirpath, dirnames, filenames) in walk(directory_of_images):
        for filename in filenames:
            accepted_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tiff')
            if filename.endswith(accepted_extensions):
                images_to_process.append(os.path.join(dirpath, filename))
        return images_to_process


def image_pixel_differences(base_image, comparison_image):
    """
    Calculates the bounding box of the non-zero regions in the
    image.

    :param base_image: target image to find
    :param comparison_image:  set of images containing the target image
    :return: The bounding box is returned as a 4-tuple defining the
             left, upper, right, and lower pixel coordinate. If the image
             is completely empty, this method returns None.
    """
    # Returns the absolute value of the pixel-by-pixel
    # difference between two images.
    diff = ImageChops.difference(base_image, comparison_image)
    if diff.getbbox():
        return False
    else:
        return True


image_directory = 'female_headshots_with_earrings'
images = get_image_files(image_directory)

target_image_name = 'jennifer_aniston_earrings.jpeg'
target_image = Image.open(f'{image_directory}/{target_image_name}')

#####################################################
# Compares all the images that are contained in the
# image_directory against the target image.
#####################################################
for image in images:
    compare_image = Image.open(image)
    results = image_pixel_differences(target_image, compare_image)
    if results is True:
        print(f'Photos with identical pixels: {target_image_name} <--> {image}')
    elif results is False:
        print(f'Photos with dissimilar pixels: {target_image_name} <--> {image}')
