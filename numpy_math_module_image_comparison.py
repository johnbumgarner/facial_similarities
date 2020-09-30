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
# Date Revised: September 27, 2020
# Revised by: John Bumgarner
#
# This Python script is designed to use modules Math and Numpy to compute the
# the Hamming distance and Structural Similarity Index (SSIM) measurements
# between two images. Similar images will similarity scores that fall within a
# specific threshold range and images that are dissimilar will have scores that
# fall outside this threshold.
#
##################################################################################

#############################################################################################
# The OS module in provides functions for interacting with the operating system.
#
# OS.walk() generate the file names in a directory tree by walking the tree.
#############################################################################################
import os
from os import walk

#############################################################################################
# The pandas module is used to generate DataFrames containing the hashing results for each
# algorithm.
#############################################################################################
import pandas as pd

######################################################################################
# The Python module Pillow is the folk of PIL, the Python Imaging Library
# reference: https://pillow.readthedocs.io/en/3.0.x/index.html
######################################################################################
# This module is used to load images
from PIL import Image

######################################################################################
# The module NumPy is the fundamental package for scientific computing with Python.
# reference: https://docs.scipy.org/doc/numpy/
######################################################################################
import numpy as np

######################################################################################
# This module provides access to the mathematical functions defined by the C standard.
# reference: https://docs.python.org/3/library/math.html
######################################################################################
from math import *


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
            accepted_extensions = ('.bmp', '.gif', '.jpg', '.jpeg', '.png', '.svg', '.tiff')
            if filename.endswith(accepted_extensions):
                images_to_process.append(os.path.join(dirpath, filename))
        return images_to_process


def get_percent_of_image_size(photo, percent):
    h, w = photo.size
    return int(h * (percent/100)), int(w * (percent/100))


def normal_curve_function(variance, segment):
    points = []
    y_sum = 0
    if variance == 0:
        return [],0
    for x in range(33, 96, 62//segment):
        y = pow(np.exp(1), - pow(x - 50, 2) / (2 * variance)) / sqrt(2 * np.pi * variance)
        points.append((x,y))
        y_sum += y
        points.reverse()
    return points, y_sum


def pre_process_images(image_one, image_two, additional_resize=False, max_image_size=1000):
    """
     This function is designed to resize the images using the Pillow module.

    :param image_one: primary image to evaluate against a secondary image
    :param image_two: secondary image to evaluate against the primary image
    :param additional_resize:
    :param max_image_size: maximum allowable image size in pixels
    :return: resized images
    """
    lower_boundary_size = (min(image_one.size[0], image_two.size[0]), min(image_one.size[1], image_two.size[1]))
    # reference: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
    # reference: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.LANCZOS
    image_one = image_one.resize(lower_boundary_size, resample=Image.LANCZOS)
    image_two = image_two.resize(lower_boundary_size, resample=Image.LANCZOS)
    if max(image_one.size) > max_image_size and additional_resize:
        resize_factor = max_image_size / max(image_one.size)
        image_one = image_one.resize((int(lower_boundary_size[0] * resize_factor),
                                      int(lower_boundary_size[1] * resize_factor)), resample=Image.LANCZOS)

        image_two = image_two.resize((int(lower_boundary_size[0] * resize_factor),
                                      int(lower_boundary_size[1] * resize_factor)), resample=Image.LANCZOS)
    return image_one, image_two


def get_ssim_similarity(image_one_name, image_two_name, window_size=7, dynamic_range=255):
    """
    The Structural Similarity Index (SSIM) is a method for measuring the similarity between two images.
    The SSIM index can be viewed as a quality measure of one of the images being compared, provided the
    other image is regarded as of perfect quality.

    :param image_one_name: primary image to evaluate against a secondary image
    :param image_two_name: secondary image to evaluate against the primary image
    :param window_size: The side-length of the sliding window used in comparison. Must be an odd value.
    :param dynamic_range: Dynamic range of the input image, specified as a positive scalar.
    The default dynamic range is 255 for images of data type uint8.
    :return: computational score and image names
    """
    image_one = Image.open(image_one_name)
    image_two = Image.open(image_two_name)

    if min(list(image_one.size) + list(image_two.size)) < 7:
        raise Exception("One of the images was too small to process using the SSIM approach")
    image_one, image_two = pre_process_images(image_one, image_two, True)
    image_one, image_two = image_one.convert('I'), image_two.convert('I')
    c1 = (dynamic_range * 0.01) ** 2
    c2 = (dynamic_range * 0.03) ** 2
    pixel_length = window_size ** 2
    ssim = 0.0
    adjusted_width = image_one.size[0] // window_size * window_size
    adjusted_height = image_one.size[1] // window_size * window_size
    for i in range(0, adjusted_width, window_size):
        for j in range(0, adjusted_height, window_size):
            crop_box = (i, j, i + window_size, j + window_size)
            crop_box_one = image_one.crop(crop_box)
            crop_box_two = image_two.crop(crop_box)
            np_array_one, np_array_two = np.array(crop_box_one).flatten(), np.array(crop_box_two).flatten()
            np_variable_one, np_variable_two = np.var(np_array_one), np.var(np_array_two)
            np_average_one, np_average_two = np.average(np_array_one), np.average(np_array_two)
            cov = (np.sum(np_array_one * np_array_two) - (np.sum(np_array_one) *
                                                          np.sum(crop_box_two) / pixel_length)) / pixel_length
            ssim += ((2.0 * np_average_one * np_average_two + c1) * (2.0 * cov + c2)) / \
                    ((np_average_one ** 2 + np_average_two ** 2 + c1) * (np_variable_one + np_variable_two + c2))
    similarity_percent = (ssim * pixel_length / (adjusted_height * adjusted_width)) * 100
    return round(similarity_percent, 2)


def hamming_image_resizing(image_one, image_two, resize_factor):
    """
    This function is designed to resize the images using the Pillow module
    and convert these images to 32-bit signed integer pixels, which are
    converted into numpy arrays.

    :param image_one: primary image to evaluate against a secondary image
    :param image_two: secondary image to evaluate against the primary image
    :param resize_factor: individual width variable used for each resizing operation
    :return: numpy arrays for both images
    """
    # reference: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
    # reference: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.BILINEAR
    image_one = image_one.resize(get_percent_of_image_size(image_one, resize_factor), resample=Image.BILINEAR)
    image_two = image_two.resize(get_percent_of_image_size(image_two, resize_factor), resample=Image.BILINEAR)

    # convert returns a converted copy of the image
    # the convert('I') mode returns a 32-bit signed integer pixels
    # reference: https://pillow.readthedocs.io/en/4.2.x/reference/Image.html#PIL.Image.Image.convert
    # reference: https://pillow.readthedocs.io/en/4.2.x/handbook/concepts.html#concept-modes
    image_one = image_one.convert('I')
    image_two = image_two.convert('I')
    np_image_one = np.array(image_one)
    np_image_two = np.array(image_two)
    return np_image_one, np_image_two


def get_hamming_similarity(image_one_name, image_two_name):
    """
    The Hamming distance is a method that can be used to measure the similarity between two images.
    Given two (normally binary) vectors, the Hamming distance measures the number of 'disagreements'
    between the two vectors. Two identical vectors would have zero disagreements, and thus perfect
    similarity.

    :param image_one_name: primary image to evaluate against a secondary image
    :param image_two_name: secondary image to evaluate against the primary image
    :return: computational score and image names
    '"""
    image_one = Image.open(image_one_name)
    image_two = Image.open(image_two_name)
    original_size = min(image_one.size[0], image_two.size[0]) * min(image_one.size[1], image_two.size[1])
    image_one, image_two = pre_process_images(image_one, image_two)
    assert image_one.size[0] * image_one.size[1] == original_size
    cumulative_similarity_score = 0
    sample_points, sample_sum = normal_curve_function(300, 10)
    for (resize_factor, factor_weightage) in sample_points:
        np_image_one, np_image_two = hamming_image_resizing(image_one, image_two, resize_factor)
        if (np_image_one.size / original_size < 0.1) or (np_image_two.size / original_size < 0.1):
            for (x, y) in sample_points:
                if x <= resize_factor:
                    sample_sum -= y
                    sample_points.remove((x, y))
        else:
            np_gradient_one = np.diff(np_image_one) > 1
            np_gradient_two = np.diff(np_image_two) > 1
            current_similarity_score = (np.count_nonzero(np.logical_not(np.logical_xor(np_gradient_one, np_gradient_two))) / np_gradient_one.size)

            weighted_similarity_score = factor_weightage * current_similarity_score
            cumulative_similarity_score += weighted_similarity_score
    average_similarity_score = (cumulative_similarity_score / sample_sum) * 100
    return round(average_similarity_score, 2)


# pandas DataFrames used for the algorithm hashing results
df_ssim_results = pd.DataFrame(columns=['base_image', 'comparison_image', 'similarity score'])
df_hamming_results = pd.DataFrame(columns=['base_image', 'comparison_image', 'similarity score'])

target_image = 'jennifer_aniston.jpeg'
image_directory = 'female_headshots_with_earrings'

images = get_image_files(image_directory)

for image in images:
    ssim_result = get_ssim_similarity(target_image, image)
    df_ssim_results = df_ssim_results.append({'base_image': target_image,
                                              'comparison_image': image.split('/')[1],
                                              'similarity score': ssim_result}, ignore_index=True)

    hamming_result = get_hamming_similarity(target_image, image)
    df_hamming_results = df_hamming_results.append({'base_image': target_image,
                                                    'comparison_image': image.split('/')[1],
                                                    'similarity score': hamming_result}, ignore_index=True)


# Display the results in a pandas DataFrame, which is sorted by similarity score
# the index numbers are removed from this output
final_df = df_ssim_results.sort_values(by=['similarity score'], ascending=False)
print(final_df.to_string(index=False))
