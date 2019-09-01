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

#########################################################################################################################################
# This Python module was developed by Johannes Bucher
# source: https://github.com/JohannesBuchner/imagehash
#
# The module has 4 hashing methods:
# 1. aHash - average hash, for each of the pixels output 1 if the pixel is bigger or equal to the average and 0 otherwise.
#
# 2. pHash - perceptive hash, does the same as aHash, but first it does a Discrete Cosine Transformation
#
# 3. dHash - gradient hash, calculate the difference for each of the pixel and compares the difference with the average differences.
#
# 4. wavelet - wavelet hashing, works in the frequency domain as pHash but it uses Discrete Wavelet Transformation (DWT) instead of DCT
#
# aHash, pHash and dHash use the same approach:
# 1. Scale an image into a grayscale 8x8 image
# 2. Performs some calculations for each of these 64 pixels and assigns a binary 1 or 0 value. These 64 bits form the output of algorithm
#
# Similar images will have a difference up to 6–8 bits.
#
#########################################################################################################################################
import imagehash

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


def compute_average_hash_computational_score(base_image, compare_image):
    '''
    Calculates the average hash (aHash) for 2 given images.  The aHash (also called Mean Hash) algorithm
    crunches the an image into a grayscale 8x8 image and sets the 64 bits in the hash based on whether
    the pixel's value is greater than the average color for the image.
    :param base_image: primary image to evaluate against the comparison images
    :param compare_image: images used in the aHash comparison process
    :return: computational score and image names
    '''

    hash0 = imagehash.average_hash(Image.open(f'{base_image}'))
    hash1 = imagehash.average_hash(Image.open(f'{compare_image}'))

    computational_score = (hash0 - hash1) /len(hash0.hash) ** 2
    if computational_score == float('0.0'):
        return (f'identical images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')
    elif computational_score <= float('0.20'):
        return (f'similar images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')
    elif computational_score > float('0.20'):
        return (f'dissimilar images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')


def compute_difference_hash_computational_score(base_image, compare_image):
    '''
    Calculates the difference hashing (dHash) for 2 given images. The dHash algorithm is nearly identical to aHash, but it
    tracks gradients instead of evaluating frequency patterns like aHash.
    :param base_image: primary image to evaluate against the comparison images
    :param compare_image: images used in the dHash comparison process
    :return: computational score and image names
    '''
    hash0 = imagehash.dhash(Image.open(f'{base_image}'))
    hash1 = imagehash.dhash(Image.open(f'{compare_image}'))

    computational_score = (hash0 - hash1) / len(hash0.hash) ** 2
    if computational_score == float('0.0'):
        return (f'identical images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')
    elif computational_score <= float('0.35'):
        return (f'similar images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')
    elif computational_score > float('0.35'):
        return (f'dissimilar images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')

def compute_perception_hash_computational_score(base_image, compare_image):
    '''
    Calculates the perception hashing (pHash) for 2 given images.  The phash algorithm is similar to aHash algorithm
    but use a discrete cosine transform (DCT) and compares based on frequencies rather than color values.
    :param base_image: primary image to evaluate against the comparison images
    :param compare_image: images used in the pHash comparison process
    :return: computational score and image names
    '''
    hash0 = imagehash.phash(Image.open(f'{base_image}'))
    hash1 = imagehash.phash(Image.open(f'{compare_image}'))

    computational_score = (hash0 - hash1) /len(hash0.hash) ** 2
    if computational_score == float('0.0'):
        return (f'identical images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')
    elif computational_score <= float('0.35'):
        return (f'similar images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')
    elif computational_score > float('0.35'):
        return (f'dissimilar images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')

def compute_wavelet_hash_computational_score(base_image, compare_image):
    '''
    Calculates the wavelet hashing (wHash) for 2 given images.  The wHash algorithm uses Discrete Wavelet Transform to transform image
    pixels into wavelets, which are then used for wavelet-based compression and coding.
    :param base_image: primary image to evaluate against the comparison images
    :param compare_image: images used in the (wHash comparison process
    :return: computational score and image names
    '''
    hash0 = imagehash.whash(Image.open(f'{base_image}'))
    hash1 = imagehash.whash(Image.open(f'{compare_image}'))

    computational_score = (hash0 - hash1) / len(hash0.hash) ** 2
    if computational_score == float('0.0'):
        return (f'identical images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')
    elif computational_score <= float('0.15'):
        return (f'similar images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')
    elif computational_score > float('0.15'):
        return (f'dissimilar images. similarity score: {computational_score} images evaluated: {base_image} -- {compare_image}')


average_hash_results = []
difference_hash_results = []
perception_hash_results = []
wavelet_hash_results = []

target_image  = 'jennifer_aniston.jpeg'
image_directory = 'female_headshots_with_earrings'

images = get_image_files(image_directory)

for image in images:

    ahash_results = compute_average_hash_computational_score(target_image, image)
    average_hash_results.append(ahash_results)

    dhash_results = compute_difference_hash_computational_score(target_image, image)
    difference_hash_results.append(dhash_results)

    phash_results = compute_perception_hash_computational_score(target_image, image)
    perception_hash_results.append(phash_results)

    whash_results = compute_wavelet_hash_computational_score(target_image, image)
    wavelet_hash_results.append(whash_results)

for item in sorted(average_hash_results, reverse=True):
    print (item)