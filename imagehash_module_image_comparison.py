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
# This Python script is designed to use the ImageHash module to calculate the
# variance difference between two images. If these images have similarity
# score of 0 then the images are identical. Similar images will similarity scores
# that fall within a specific threshold range and images that are dissimilar will
# have scores that fall outside this threshold.
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

#######################################################################################################################
# This Python module was developed by Johannes Bucher
# source: https://github.com/JohannesBuchner/imagehash
#
# The module has 4 hashing methods:
#
# 1. aHash - average hash, for each of the pixels output 1 if the pixel is bigger or equal to the average and
# 0 otherwise.
#
# 2. pHash - perceptive hash, does the same as aHash, but first it does a Discrete Cosine Transformation
#
# 3. dHash - gradient hash, calculate the difference for each of the pixel and compares the difference with the
# average differences.
#
# 4. wavelet - wavelet hashing, works in the frequency domain as pHash but it uses Discrete Wavelet Transformation
# (DWT) instead of DCT
#
# aHash, pHash and dHash all use the same approach:
# 1. Scale an image into a grayscale 8x8 image
# 2. Performs some calculations for each of these 64 pixels and assigns a binary 1 or 0 value. These 64 bits form
# the output of algorithm
#

#######################################################################################################################
import imagehash


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


def compute_average_hash_computational_score(base_image, comparison_image):
    """
    Calculates the average hash (aHash) for 2 given images.  The aHash (also called Mean Hash) algorithm
    crunches the an image into a grayscale 8x8 image and sets the 64 bits in the hash based on whether
    the pixel's value is greater than the average color for the image.
    :param base_image: primary image to evaluate against the comparison images
    :param comparison_image: images used in the aHash comparison process
    :return: computational score and image names
    """
    # load the images and compute their average hashes
    hash0 = imagehash.average_hash(Image.open(base_image))
    hash1 = imagehash.average_hash(Image.open(comparison_image))

    # Methods to generate computational score:
    #
    # Method 1: computational_score = (hash0 - hash1) / len(hash0.hash) **2
    # this method generate a computational_score as a percentage, such as 0.125000
    # this percentage can be multiplied (.125000 x 100) to obtain the value of 12.50%
    # which can be rounded to 13%, which is the average value per hash bit.
    #
    # Method 2: computational_score = (hash0 - hash1)
    # this method generate a computational_score as an int, such as 8.
    #
    computational_score = (hash0 - hash1)
    return computational_score


def compute_perception_hash_computational_score(base_image, comparison_image):
    """
    Calculates the perception hashing (pHash) for 2 given images.  The phash algorithm is similar to aHash algorithm
    but use a discrete cosine transform (DCT) and compares based on frequencies rather than color values.
    :param base_image: primary image to evaluate against the comparison images
    :param comparison_image: images used in the pHash comparison process
    :return: computational score and image names
    """
    # load the images and compute their perception hashes
    hash0 = imagehash.phash(Image.open(base_image))
    hash1 = imagehash.phash(Image.open(comparison_image))

    # Methods to generate computational score:
    #
    # Method 1: computational_score = (hash0 - hash1) / len(hash0.hash) **2
    # this method generate a computational_score as a percentage, such as 0.125000
    # this percentage can be multiplied (.125000 x 100) to obtain the value of 12.50%
    # which can be rounded to 13%, which is the perception value per hash bit.
    #
    # Method 2: computational_score = (hash0 - hash1)
    # this method generate a computational_score as an int, such as 8.
    #
    computational_score = (hash0 - hash1)
    return computational_score


def compute_difference_hash_computational_score(base_image, comparison_image):
    """
    Calculates the difference hashing (dHash) for 2 given images. The dHash algorithm is nearly identical to aHash, but it
    tracks gradients instead of evaluating frequency patterns like aHash.
    :param base_image: primary image to evaluate against the comparison images
    :param comparison_image: images used in the dHash comparison process
    :return: computational score and image names
    """
    # load the images and compute their difference hashes
    hash0 = imagehash.dhash(Image.open(base_image))
    hash1 = imagehash.dhash(Image.open(comparison_image))

    # Methods to generate computational score:
    #
    # Method 1: computational_score = (hash0 - hash1) / len(hash0.hash) **2
    # this method generate a computational_score as a percentage, such as 0.125000
    # this percentage can be multiplied (.125000 x 100) to obtain the value of 12.50%
    # which can be rounded to 13%, which is the difference value per hash bit.
    #
    # Method 2: computational_score = (hash0 - hash1)
    # this method generate a computational_score as an int, such as 8.
    #
    computational_score = (hash0 - hash1)
    return computational_score


def compute_wavelet_hash_computational_score(base_image, comparison_image):
    """
    Calculates the wavelet hashing (wHash) for 2 given images.  The wHash algorithm uses Discrete Wavelet Transform to transform image
    pixels into wavelets, which are then used for wavelet-based compression and coding.
    :param base_image: primary image to evaluate against the comparison images
    :param comparison_image: images used in the (wHash comparison process
    :return: computational score and image names
    """
    hash0 = imagehash.whash(Image.open(base_image))
    hash1 = imagehash.whash(Image.open(comparison_image))
    # Methods to generate computational score:
    #
    # Method 1: computational_score = (hash0 - hash1) / len(hash0.hash) **2
    # this method generate a computational_score as a percentage, such as 0.125000
    # this percentage can be multiplied (.125000 x 100) to obtain the value of 12.50%
    # which can be rounded to 13%, which is the wavelet value per hash bit.
    #
    # Method 2: computational_score = (hash0 - hash1)
    # this method generate a computational_score as an int, such as 8.
    #
    computational_score = (hash0 - hash1)
    return computational_score


# pandas DataFrames used for the algorithm hashing results
df_average_hash = pd.DataFrame(columns=['base_image', 'comparison_image', 'similarity score'])
df_perception_hash = pd.DataFrame(columns=['base_image', 'comparison_image', 'similarity score'])
df_difference_hash = pd.DataFrame(columns=['base_image', 'comparison_image', 'similarity score'])
df_wavelet_hash = pd.DataFrame(columns=['base_image', 'comparison_image', 'similarity score'])

# options to disable certain display limits
pd.options.display.max_columns = None
pd.options.display.max_rows = None

target_image = 'jennifer_aniston.jpeg'
image_directory = 'female_headshots_with_earrings'

images = get_image_files(image_directory)

for image in images:

    ahash_result = compute_average_hash_computational_score(target_image, image)
    df_average_hash = df_average_hash.append({'base_image': target_image, 'comparison_image': image.split('/')[1],
                                              'similarity score': ahash_result}, ignore_index=True)

    phash_result = compute_perception_hash_computational_score(target_image, image)
    df_perception_hash = df_perception_hash.append({'base_image': target_image, 'comparison_image': image.split('/')[1],
                                                    'similarity score': phash_result}, ignore_index=True)

    dhash_result = compute_difference_hash_computational_score(target_image, image)
    df_difference_hash = df_difference_hash.append({'base_image': target_image, 'comparison_image': image.split('/')[1],
                                                    'similarity score': dhash_result}, ignore_index=True)

    whash_result = compute_wavelet_hash_computational_score(target_image, image)
    df_wavelet_hash = df_wavelet_hash.append({'base_image': target_image, 'comparison_image': image.split('/')[1],
                                              'similarity score': whash_result}, ignore_index=True)


# Display the results in a pandas DataFrame, which is sorted by similarity score
# the index numbers are removed from this output
final_df = df_difference_hash.sort_values(by=['similarity score'], ascending=True)
print(final_df.to_string(index=False))
