######################################################################################
# The Python module Pillow is the folk of PIL, the Python Imaging Library
# reference: https://pillow.readthedocs.io/en/3.0.x/index.html
######################################################################################
# This module is used to load images
from PIL import Image
# This module contains a number of arithmetical image operations
from PIL import ImageChops

# This list contains images of well-known female actresses
# wearing earrings
headshots = ['elizabeth hurley_earrings.jpeg', 'hilary_swank_earrings.jpeg', 'jennifer_anoston_earrings.jpeg', 'jennifer_anoston_earrings_02.jpeg',
             'jennifer_anoston_earrings_03.jpeg', 'jennifer_garner_earrings.jpeg', 'julia_roberts_earrings.jpeg', 'maggie_gyllenhaal_earrings.jpg',
             'natalie_portman_earrings.jpeg', 'nicole_kidman_earrings.jpeg', 'poppy_delevingne_earrings.jpeg','taylor_swift_earrings.jpeg']

image_directory = 'female_headshots_with_earrings'
base_image_name = 'jennifer_anoston_earrings_02.jpeg'
base_image = Image.open(f'{image_directory}/{base_image_name}')

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

# Evaluate all the images that are contained
# in the list headshots against the target image
for image in headshots:
    compare_image = Image.open(f'{image_directory}/{image}')
    results = image_pixel_differences(base_image, compare_image)
    if results == True:
        print(f'These images have identical pixels: {base_image_name}, {image}')
    elif results == False:
        print (f'These images have dissimilar pixels: {base_image_name}, {image}')