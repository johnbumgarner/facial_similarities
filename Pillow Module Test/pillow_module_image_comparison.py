#########################################
# The Python module Pillow is the
# folk of PIL (Python Imaging Library)
#########################################
# this module is used to load images
from PIL import Image
# This module contains a number of arithmetical image operations,
from PIL import ImageChops

# This list contains multiple images of Idris Elba, who
# was named the 2018 Sexiest Man Alive by People magazine.
idris_elba_headshots = ['idris_elba_headshot_01.jpeg', 'idris_elba_headshot_02.jpeg',
                        'idris_elba_headshot_03.jpeg', 'idris_elba_headshot_04.jpeg',
                        'idris_elba_headshot_05.jpeg', 'idris_elba_headshot_06.jpeg',
                        'idris_elba_headshot_07.jpeg', 'idris_elba_headshot_08.jpeg',
                        'idris_elba_headshot_09.jpeg', 'idris_elba_headshot_10.jpeg']


image_directory = 'idris_elba_headshots'
base_image_name = 'idris_elba_headshot_05.jpeg'
base_image = Image.open(f'{image_directory}/{base_image_name}')

def image_pixel_differences(base_image, compare_image):
    """
    Calculates the bounding box of the non-zero regions in the
    image.

    :param base_image:
    :param compare_image:
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

# Evaluate all images that are contained
# in the list idris_elba_headshots
for image in idris_elba_headshots:
    compare_image = Image.open(f'{image_directory}/{image}')
    results = image_pixel_differences(base_image, compare_image)
    if results == True:
        print(f'These images have identical pixels: {base_image_name}, {image}')
    elif results == False:
        print (f'These images have dissimilar pixels: {base_image_name}, {image}')
