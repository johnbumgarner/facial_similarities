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

######################################################################################
# This module provides functions that create iterators for efficient looping
# reference: https://docs.python.org/2/library/itertools.html
######################################################################################
from itertools import combinations_with_replacement

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

def get_percent_of_image_size(image, percent):
    h, w = image.size
    return (int(h * (percent/100)), int(w * (percent/100)))

def filter_by_threshold(imgsWithPercentList, threshold):
    filteredList = []
    for obj in imgsWithPercentList:
        image_one, image_two , percent = obj
        if int(percent) >= threshold:
            filteredList.append((image_one, image_two, percent))
    return filteredList

def normal_curve_function(variance , segment):
    points = []
    y_sum = 0
    if variance == 0:
        return [],0
    for x in range(33, 96, 62//segment):
        y = pow(np.exp(1), - pow(x - 50, 2) / (2 * variance) ) /sqrt(2 * np.pi * variance)
        points.append((x,y))
        y_sum += y
        points.reverse()
    return points, y_sum

def pre_process_images(image_one, image_two, additionalResize = False, maxImageSize = 1000):
    lowerBoundSize = (min(image_one.size[0], image_two.size[0]), min(image_one.size[1], image_two.size[1]))
    image_one = image_one.resize(lowerBoundSize, resample=Image.LANCZOS)
    image_two = image_two.resize(lowerBoundSize, resample=Image.LANCZOS)
    if max(image_one.size) > maxImageSize and additionalResize:
        resizeFactor = maxImageSize / max(image_one.size)
        image_one = image_one.resize((int (lowerBoundSize[0] * resizeFactor), int (lowerBoundSize[1] * resizeFactor)), resample=Image.LANCZOS)
        image_two = image_two.resize((int (lowerBoundSize[0] * resizeFactor), int (lowerBoundSize[1] * resizeFactor)), resample=Image.LANCZOS)
    return image_one, image_two

def get_ssim_similarity(image_one_name, image_two_name, windowSize = 7, dynamicRange = 255):
    '''
    The Structural Similarity Index (SSIM) is a method for measuring the similarity between two images.
    The SSIM index can be viewed as a quality measure of one of the images being compared, provided the
    other image is regarded as of perfect quality.

    :param image_one_name: primary image to evaluate against a secondary image
    :param image_two_name: secondary image to evaluate against the primary image
    :param windowSize:
    :param dynamicRange:
    :return:  computational score and image names
    '''
    image_one = Image.open(f'{image_one_name}')
    image_two = Image.open(f'{image_two_name}')

    if min(list(image_one.size) + list(image_two.size)) < 7:
        raise Exception ("One of the images was too small to process uing the SSIM approach")
    image_one, image_two = pre_process_images(image_one, image_two, True)
    image_one, image_two = image_one.convert('I'), image_two.convert('I')
    c1 = (dynamicRange * 0.01) ** 2
    c2 = (dynamicRange * 0.03) ** 2
    pixelLength = windowSize ** 2
    ssim = 0.0
    adjustedWidth = image_one.size[0] // windowSize * windowSize
    adjustedHeight = image_one.size[1] // windowSize * windowSize
    for i in range(0, adjustedWidth, windowSize):
        for j in range(0, adjustedHeight, windowSize):
            cropBox = (i, j, i + windowSize, j + windowSize)
            window1 = image_one.crop(cropBox)
            window2 = image_two.crop(cropBox)
            npWindow1, npWindow2 = np.array(window1).flatten(), np.array(window2).flatten()
            npVar1, npVar2 = np.var(npWindow1), np.var(npWindow2)
            npAvg1, npAvg2 = np.average(npWindow1), np.average(npWindow2)
            cov = (np.sum(npWindow1 * npWindow2) - (np.sum(npWindow1) * np.sum(window2) / pixelLength)) / pixelLength
            ssim += ((2.0 * npAvg1 * npAvg2 + c1) * (2.0 * cov + c2)) / ((npAvg1 ** 2 + npAvg2 ** 2 + c1) * (npVar1 + npVar2 + c2))
    similarityPercent = (ssim * pixelLength / (adjustedHeight * adjustedWidth)) * 100
    return (image_one_name, image_two_name, round(similarityPercent, 2))

def hamming_image_resizing(image_one, image_two, resizeFactor):
    image_one = image_one.resize(get_percent_of_image_size(image_one, resizeFactor), resample=Image.BILINEAR)
    image_two = image_two.resize(get_percent_of_image_size(image_two, resizeFactor), resample=Image.BILINEAR)
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
    '''
    The Hamming distance is a method that can be used to measure the similarity between two images.
    Given two (normally binary) vectors, the Hamming distance measures the number of 'disagreements'
    between the two vectors. Two identical vectors would have zero disagreements, and thus perfect
    similarity.

    :param image_one_name: primary image to evaluate against a secondary image
    :param image_two_name: secondary image to evaluate against the primary image
    :return: computational score and image names
    '''
    image_one = Image.open(f'{image_one_name}')
    image_two = Image.open(f'{image_two_name}')
    origSize = min(image_one.size[0], image_two.size[0]) * min(image_one.size[1], image_two.size[1])
    image_one, image_two = pre_process_images(image_one, image_two)
    assert image_one.size[0] * image_one.size[1] == origSize
    cumulativeSimilarityScore = 0
    samplePts, sampleSum = normal_curve_function(300, 10)
    for (resizeFactor, factorWeightage) in samplePts:
        np_image_one, np_image_two = hamming_image_resizing(image_one, image_two, resizeFactor)
        if (np_image_one.size / origSize < 0.1) or (np_image_two.size / origSize < 0.1):
            for (x, y) in samplePts:
                if x <= resizeFactor:
                    sampleSum -= y
                    samplePts.remove((x, y))
        else:
            np_gradient_one = np.diff(np_image_one) > 1
            np_gradient_two = np.diff(np_image_two) > 1
            currentSimilarityScore = (np.count_nonzero(np.logical_not(np.logical_xor(np_gradient_one, np_gradient_two)))/np_gradient_one.size)
            weightedSimilarityScore = factorWeightage * currentSimilarityScore
            cumulativeSimilarityScore += weightedSimilarityScore
    averageSimilarityScore = (cumulativeSimilarityScore / sampleSum) * 100
    return image_one_name, image_two_name, round(averageSimilarityScore, 2)


def process_image_similarity(image_filenames, approach, threshold):
    params = combinations_with_replacement(image_filenames, 2)
    similarity_percentage_ratio = []

    if approach == 'HAM':
        for (image_one, image_two) in params:
            result = get_hamming_similarity(image_one, image_two)
            similarity_percentage_ratio.append(result)
    elif approach == 'SIM':
        for (image_one, image_two) in params:
            result = get_ssim_similarity(image_one, image_two)
            similarity_percentage_ratio.append(result)

    final_results = filter_by_threshold(similarity_percentage_ratio, threshold)
    return final_results


target_image  = 'jennifer_aniston.jpeg'
image_directory = 'female_headshots_with_earrings'

images = get_image_files(image_directory)

ssim_results = []
hamming_results = []

for image in images:
    ssim = get_ssim_similarity(target_image, image)
    ssim_results.append(ssim)

    hamming = get_hamming_similarity(target_image, image)
    hamming_results.append(hamming)

ssim_results.sort(key = lambda tup: tup[2:2])
print ('##################################################')
print ('STRUCTURAL SIMILARITY (SSIM) RESULTS')
print ('##################################################')
for item in ssim_results:
    original_image = item[0]
    compare_image = item[1]
    computational_score = item[2]
    if computational_score == 100.0:
        print(f'identical images. similarity score: {computational_score} images evaluated: {original_image} -- {compare_image}')
    elif computational_score < 21.0:
        print(f'similar images. similarity score: {computational_score} images evaluated: {original_image} -- {compare_image}')
    elif computational_score > 21.0:
        print(f'dissimilar images. similarity score: {computational_score} images evaluated: {original_image} -- {compare_image}')

print ('\n')

hamming_results.sort(key = lambda tup: tup[2:2])
print ('##################################################')
print ('HAMMING DISTANCE RESULTS')
print ('##################################################')
for item in hamming_results:
    original_image = item[0]
    compare_image = item[1]
    computational_score = item[2]
    if computational_score == 100.0:
        print(f'identical images. similarity score: {computational_score} images evaluated: {original_image} -- {compare_image}')
    elif computational_score < 50.0:
        print(f'similar images. similarity score: {computational_score} images evaluated: {original_image} -- {compare_image}')
    elif computational_score > 50.0:
        print(f'dissimilar images. similarity score: {computational_score} images evaluated: {original_image} -- {compare_image}')


# This section compares all the images in the control set
similarity_results = []
# HAM = Hamming distance
# SIM = Structural Similarity Index (SSIM)
# the threshold float is adjustable
results = process_image_similarity(images, 'HAM', 0.5)
for item in results:
    similarity_results.append(item)

print ('\n')

similarity_results.sort(key = lambda tup: tup[2], reverse = True)
print ('##################################################')
print ('IMAGE SIMILARITY RESULTS - ALL IMAGES')
print ('##################################################')
for item in similarity_results:
    original_image = item[0]
    compare_image = item[1]
    computational_score = item[2]
    if computational_score == 100.0:
        print(f'identical images. similarity score: {computational_score} images evaluated: {original_image} -- {compare_image}')
    elif computational_score < 50.0:
        print(f'similar images. similarity score: {computational_score} images evaluated: {original_image} -- {compare_image}')
    elif computational_score > 50.0:
        print(f'dissimilar images. similarity score: {computational_score} images evaluated: {original_image} -- {compare_image}')