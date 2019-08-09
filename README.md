# Image comparison experiments

This repository contains Python code used for various image simarility experiments. The primary purpose of these tests was to determine the capabilities and limitations of individual Python modules and various techniques used for image simarility. 

The primary objective of these experiments: Provided a photo of a target person, and a set of other photos, is the target person one of the people in the set of photographs. 

The set of images used in these image simarility experiments were of well-known female actresses wearing earrings.  

![Females_with_earrings](https://github.com/johnbumgarner/image_simarility_experiments/blob/master/females_with_earrings_test_images.jpg)


## Image Simarility Experiments

**Experiment 1**

This experiment used the Python module Pillow, which a the folk of PIL, the Python Imaging Library. The function used was PIL.ImageChops.difference(image1, image2), which returns the absolute value of the pixel-by-pixel difference between the two images.








