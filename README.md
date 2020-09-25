<p align="center">
  <img src="https://github.com/johnbumgarner/image_closeness_experiments/blob/master/graphic/facial_recognition.png"/>
</p>

# Overview Image Similarities

<p align="justify">

Most humans can look at two photos and quickly determine if the images are either similarity or dissimilarity in nature. Computers can be programmed to perform a similar task, but the results can vary, because of multiple factors(e.g., lighting conditions, perspectives) that humans can instinctively do automatically.  Humans have little difficulty seeing the subtle differents between a rose and a camellia or a gardenia and a rose. A computer on the other hand will recognize these objects as flowers, but would likely classify all of these images as a single type of flower, roses.   

There are numerous use cases for image similarities technologies. These use cases range from duplicate image detection to domain specific image clustering. Identifying duplicate images in Apple Photo is a common use case for many of us dealing with a large digital image library. Some of us have likey used Google’s Reverse Image Search to look for a specific photo that we want to know more about. Google will scour its massive database for images similar to the one used in your query. 

</p>

## Primary objective of this repository
<p align="justify">
This repository is going to examine several of the methods used to ascertain if two or more images have similarity or dissimilarity. The set of images used in these image simarility tests are publicly available photographs of well-known female actresses. One dataset has 12 images of these actresses wearing earrings. The different photos of Jennifer Aniston are within the first dataset.  The second dataset consists of 50 images of actresses with no duplicates.  The second set is more diversed, because it included mutiple skin tones and hair colors.   

Another objective of this repository is to determine the capabilities and limitations of the Python libraries used to perform these image simarility tests.
</p>

## Image Simarility Experiments

### Python Imaging Library:

<p align="justify">
This experiment used the Python module <i>Pillow</i>, which is a folk of PIL, the Python Imaging Library. The Pillow function used in this experiment was 
<i>PIL.ImageChops</i>. The ImageChops module contains a number of arithmetical image operations, called channel operations (“chops”). These can be used for various purposes, including special effects, image compositions, algorithmic painting, and more.  The sub-function used was <i>difference</i>, which returns the absolute value of the pixel-by-pixel difference between two images.  Here is how the function is called.
  
    PIL.ImageChops.difference(base_image, comparison_image)
  
The base_image in this experiment was one of Jennifer Aniston. The comparison_image dataset used in this experiment was the one that contained 12 images of actresses wearing earrings.  The are 3 images of Jennifer Aniston in this dataset, but only one of these images is an absolute match.  

    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/elizabeth hurley_earrings.jpeg
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/poppy_delevingne_earrings.jpeg
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/hilary_swank_earrings.jpeg
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/nicole_kidman_earrings.jpeg
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/jennifer_aniston_earrings_03.jpeg
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/jennifer_aniston_earrings_02.jpeg
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/jennifer_garner_earrings.jpeg
    Photos with identical pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/jennifer_aniston_earrings.jpeg<
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/taylor_swift_earrings.jpeg
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/maggie_gyllenhaal_earrings.jpg
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/julia_roberts_earrings.jpeg
    Photos with dissimilar pixels: jennifer_aniston_earrings.jpeg <--> female_headshots_with_earrings/natalie_portman_earrings.jpeg

This pixel-by-pixel comparison is useful in finding exact duplicates, but it will not match images that have been slightly altered thus resulting in a different pixel value.
</p>

### ImageHash Library:
This experiment used the Python module <i>ImageHash,/i>, which was developed by Johannes Bucher.  This module has four hashing methods:

1. aHash: average hash, for each of the pixels output 1 if the pixel is bigger or equal to the average and 0 otherwise.

2. pHash: perceptive hash, does the same as aHash, but first it does a Discrete Cosine Transformation (DCT).

3. dHash:  gradient hash, calculate the difference for each of the pixel and compares the difference with the average differences.

4. wavelet: wavelet hashing, works in the frequency domain as pHash but it uses Discrete Wavelet Transformation (DWT) instead of DCT.

#### aHash


All four hashing methods were able to accurately identify the two images of Jennifer Aniston.  All the hashing methods produce similarity scores.  The threshold of these scores are adjustable.  

During testing the threshold for _aHash_ similar images was set at less than 20, which successful matched another Jennifer Aniston image (jennifer_anoston_earrings_03.jpeg), but produced 2 false positives.  The _aHash_ dissimilar image threshold was set at greater than 20.  The third Jennifer Aniston (jennifer_anoston_earrings.jpeg) was in this dissimilar set.  



The threshold for _dHash_ similar images was set at less than 35, which successful matched another Jennifer Aniston image (jennifer_anoston_earrings.jpeg) and produced no false positives. The _dHash_ dissimilar image threshold was set at greater than 35.  The third Jennifer Aniston (jennifer_anoston_earrings_03.jpeg) was in this dissimilar set. 



The threshold for _pHash_ similar images was set at less than 35, which did not match any addtional images of Jennifer Aniston in the control set. Setting this threshold to less than 40 produced 2 false positives. The _pHash_ dissimilar image threshold was set at greater than 40.  Both of the other Jennifer Aniston images were in this dissimilar set. 



The threshold for _wavelet_ similar images was set at less than 15, which did not match any additional images of Jennifer Aniston in the control set. Setting this threshold to less than 20 produced 2 false positives. Setting this threshold to less than 30 produced 3 false positives and identified another Jennifer Aniston (jennifer_anoston_earrings_03.jpeg) image. The _wavelet_ dissimilar image threshold was set at greater than 15.  Both of the other Jennifer Aniston images were in this dissimilar set. 
</p>

### Numpy and Math Library:

<p align="justify">
This experiment used the Python modules Numpy and Math. Numpy is one of fundamental packages for scientific computing with Python and Math provides access to various access mathematical functions. This experiment focused on Structural Similarity Index (SSIM), which is the a method for measuring the similarity between two images and Hamming distance, which determines how similar two images are.
</p>

<p align="justify">
In testing both methods were ables to successfully identify the target image of Jennifer Aniston from the control set of images. As in the previous experiments the SSIM and Hamming measurement generated a computational score based on similarity and distance. Establishing a computational score threshold using these two measurement methods was highly problematic, because it produced a considerable amount of false positives.
</p>



## Facial Detection/Recognition/Prediction Experiments

### Experiment 1:

<p align="justify">
This experiment used the Python module cv2. The Python package is opencv-python.  This opencv-python package is the Open Source Computer Vision (OpenCV), which is a computer vision and machine learning software library. Computer vision and digital image processing are currently being widely applied in face recognition, criminal investigation, signature pattern detection in banking, digital documents analysis and smart tag based vehicles for recognition. 
</p>

<p align="justify">
The LBPH face recognition algorithm was used in this experiment. Local Binary Pattern (LBP) is a simple and efficient texture operator which labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number. The experiment also uses the Haar Cascade, which is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of features.
</p>

The Haar Cascade classifiers used in this experiment were:

1. haarcascade_frontalface_default.xml 
2. haarcascade_eye.xml
3. haarcascade_mcs_nose.xml
4. haarcascade_smile.xml

<p align="justify">
During testing it was noted that all these Haar Cascade classifiers were temperamental and required continually tuning of the parameters scaleFactor and minNeighbors used by detectMultiScale.  The angle of the faces within the images were also a key factor when detecting facial features. Images containing direct frontal faces produced the best results as shown below.
</p>

<p align="center"> <br>
<img src="https://github.com/johnbumgarner/image_simarility_experiments/blob/master/haar_cascade_features.jpg">
</p>

### Experiment 2:

<p align="justify">
This experiment also uses the Python module cv2 and the LBPH face recognition algorithm.  The previous experiment was designed to do some base level facial prediction. In this experiment a set of 74 images of female actresses was used to create the facial prediction training data, the associated labels and identification numbers. 
</p>

<p align="justify">
The predict feature of LBPHFaceRecognizer was utilized in this experiment. The prediction of the LBPH face recognition algorithm will analyze an image containing face and attempt to identify the face contain in the image against images within the training data. If a probable match is found within the training data then the person name and the confidence score will be overlaid on the image displayed as shown below:
</p>

<p align="center"><br>
<img src="https://github.com/johnbumgarner/image_similarity_experiments/blob/master/aishwarya_rai_confidence_score.jpg">
</p>

<p align="justify">
The algorithm was able to successfully predict that an image contained in the training data matched an image that was used in creating the training data. A confidence_score of '0' is a 100% match. The success rate was less accurate for images not used in the training data. 
</p>

### Experiment 3:

<p align="justify">
Like the previous experiment this one also uses the Python module cv2 and the LBPH face recognition algorithm, but this experiment includes horizontal image flipping. This experiment also used the same set of 74 images of female actresses. These images were used to create the facial prediction training data, the associated labels and identification numbers.
</p>

<p align="justify">
The predict feature of LBPHFaceRecognizer was also utilized in this experiment. The prediction of the LBPH face recognition algorithm will analyze an image containing face and attempt to identify the face contain in the image against images (original and horizontal flipped) within the training data. If a probable match is found within the training data then the person name and the confidence score will be overlaid on the image displayed as shown below:
</p>

<p align="center"><br>
<img src="https://github.com/johnbumgarner/image_similarity_experiments/blob/master/flip_jennifer_aniston_confidence_score.jpg">
</p>

<p align="justify">
The algorithm was able to successfully predict that an image contained in the training data matched either an original or horizontal image that was used in creating the training data. The success rate was again less accurate for images not used in the training data. 
</p>

### Notes:

_The code within this repository is **not** production ready. It was **strictly** designed for experimental testing purposes only._


####
<p align="center"><br>
<img src="https://github.com/johnbumgarner/image_simarility_experiments/blob/master/females_with_earrings_test_images.jpg">
</p>

In Euclidean geometry, two objects are similar if they have the same shape, or one has the same shape as the mirror image of the other. More precisely, one can be obtained from the other by uniformly scaling, possibly with additional translation, rotation and reflection.

```python
s = "Python syntax highlighting"
print s
```

