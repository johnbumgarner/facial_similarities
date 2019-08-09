######################################################################################
# The Python module Pillow is the folk of PIL, the Python Imaging Library
# reference: https://pillow.readthedocs.io/en/3.0.x/index.html
######################################################################################
# This module is used to load images
from PIL import Image

# numpy is one of fundamental packages for scientific computing with Python
import numpy as np

# provides access to the mathematical functions
from math import *

# provides functions that create iterators for efficient looping
from itertools import combinations_with_replacement

image_directory = 'female_headshots_with_earrings'

# This list contains images of well-known female actresses
# wearing earrings
headshots = ['elizabeth hurley_earrings.jpeg', 'hilary_swank_earrings.jpeg', 'jennifer_anoston_earrings.jpeg', 'jennifer_anoston_earrings_02.jpeg',
             'jennifer_anoston_earrings_03.jpeg', 'jennifer_garner_earrings.jpeg', 'julia_roberts_earrings.jpeg', 'maggie_gyllenhaal_earrings.jpg',
             'natalie_portman_earrings.jpeg', 'nicole_kidman_earrings.jpeg', 'poppy_delevingne_earrings.jpeg','taylor_swift_earrings.jpeg']

target_image  = 'jennifer_anoston_earrings_02.jpeg'

# Image file extensions
accepted_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'JPG']

def filter_file_extensions(filenames):
    acceptable_file_types = []
    for fname in filenames:
        extension = (fname.split('.'))[-1]
        if extension in accepted_extensions:
            acceptable_file_types.append(fname)
    return acceptable_file_types

def getPercentOfImageSize(image, percent):
    h, w = image.size
    return (int(h * (percent/100)), int(w * (percent/100)))

def filterByThreshold(imgsWithPercentList, threshold):
    filteredList = []
    for obj in imgsWithPercentList:
        image_one, image_two , percent = obj
        if int(percent) >= threshold:
            filteredList.append((image_one, image_two, percent))
    return filteredList

def normalCurveFunction(variance , segment):
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

def preProcessImage(image_one, image_two, additionalResize = False, maxImageSize = 1000):
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
    image_one, image_two = Image.open(f'{image_directory}/{image_one_name}'), Image.open(f'{image_directory}/{image_two_name}')
    if min(list(image_one.size) + list(image_two.size)) < 7:
        raise Exception ("One of the images was too small to process uing the SSIM approach")
    image_one, image_two = preProcessImage(image_one, image_two, True)
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

def hammingResize(image_one, image_two, resizeFactor):
    image_one = image_one.resize(getPercentOfImageSize(image_one, resizeFactor), resample=Image.BILINEAR)
    image_two = image_two.resize(getPercentOfImageSize(image_two, resizeFactor), resample=Image.BILINEAR)
    image_one = image_one.convert('I')
    image_two = image_two.convert('I')
    np_image_one = np.array(image_one)
    np_image_two = np.array(image_two)
    return np_image_one, np_image_two

def get_hamming_similarity(image_one_name, image_two_name):
    '''

    :param image_one_name: primary image to evaluate against a secondary image
    :param image_two_name: secondary image to evaluate against the primary image
    :return: computational score and image names
    '''
    image1 = Image.open(f'{image_directory}/{image_one_name}')
    image2 = Image.open(f'{image_directory}/{image_two_name}')
    origSize = min(image1.size[0], image2.size[0]) * min(image1.size[1], image2.size[1])
    image1, image2 = preProcessImage(image1, image2)
    assert image1.size[0] * image1.size[1] == origSize
    cumulativeSimilarityScore = 0
    samplePts, sampleSum = normalCurveFunction(300, 10)
    for (resizeFactor, factorWeightage) in samplePts:
        np_image_one, np_image_two = hammingResize(image1, image2, resizeFactor)
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
    return (image_one_name, image_two_name, round(averageSimilarityScore, 2))


def processSimilarity(image_filenames, approach, threshold):
    params = combinations_with_replacement(image_filenames, 2)
    similarity_percentage_ratio = []

    if approach == 'N':
        for (image_one, image_two) in params:
            result = get_hamming_similarity(image_one, image_two)
            similarity_percentage_ratio.append(result)
    elif approach == 'E':
        for (image_one, image_two) in params:
            result = get_ssim_similarity(image_one, image_two)
            similarity_percentage_ratio.append(result)

    final_results = filterByThreshold(similarity_percentage_ratio, threshold)

    return final_results


ssim_results = []
hamming_results = []
for image in headshots:
    ssim = get_ssim_similarity(target_image, image)
    ssim_results.append(ssim)

    hamming = get_hamming_similarity(target_image, image)
    hamming_results.append(hamming)


ssim_results.sort(key=lambda tup: tup[2:2], reverse=True)
print ('##################################################')
print ('STRUCTURAL SIMILARITY (SSIM) RESULTS')
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

hamming_results.sort(key=lambda tup: tup[2:2], reverse=True)
print ('##################################################')
print ('HAMMING DISTANCE RESULTS')
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
results = processSimilarity(headshots, 'N', 0.5)
for item in results:
    similarity_results.append(item)

print ('\n')

similarity_results.sort(key=lambda tup: tup[2], reverse=True)
print ('##################################################')
print ('IMAGE SIMILARITY RESULTS - ALL IMAGES')
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




