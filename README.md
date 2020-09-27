# being updated at the moment

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
  
  
```python
PIL.ImageChops.difference(base_image, comparison_image)
```
  
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
This experiment used the Python module <i>ImageHash,</i>, which was developed by Johannes Bucher.  This module has four hashing methods:

1. <b>aHash:</b> average hash, for each of the pixels output 1 if the pixel is bigger or equal to the average and 0 otherwise.

2. <b>pHash:</b> perceptive hash, does the same as aHash, but first it does a Discrete Cosine Transformation (DCT).

3. <b>dHash:</b> difference hash, calculate the difference for each of the pixel and compares the difference with the average differences.

4. <b>wavelet:</b> wavelet hashing, works in the frequency domain as pHash but it uses Discrete Wavelet Transformation (DWT) instead of DCT.

#### aHash algorithm

The <i>average hash</i> algorithm is designed to scale the input image down to 8×8 pixels and covert this smaller image to grayscale. This changes the hash from 64 pixels (64 red, 64 green, and 64 blue) to 64 total color.  Next the algorithm averages these 64 colors and computes a mean value.  Each bit of the rescaled image is evaluated against this mean value and each bit is set either above of below this value. These measurements are used to construct a hash, which will not change even if the image is scaled or the aspect ratio changes.  The hash value will not dramatically change even when someone increasing or decreasing the brightness or contrast of the image or alters its colors.  

An image hash is used to determine the <i>hamming distance</i> between two hashes. The <i>Hamming distance</i> between two strings (hashes) of equal length is the number of positions at which these strings vary. In more technical terms, it is a measure of the minimum number of changes required to turn one string into another.

A <i>Hamming distance</i> of 0 means that two images are identical, whereas a distance of 5 or less indicates that two images are probably similar.  If the <i>Hamming distance</i> is greater than 10 then the images are most likely different. 

The basic usage of the <i>average hash</i> algorithm within <i>ImageHash</i> is:

```python
hash0 = imagehash.ahash(Image.open(base_image))
hash1 = imagehash.ahash(Image.open(comparison_image))
computational_score = (hash0 - hash1)
```

The average hash algorithm correctly matched the Jennifer Aniston base image to the same Jennifer Aniston comparison image within the dataset.  The algorithm did not find any similarities between the Jennifer Aniston base image and the other Jennifer Aniston comparison images within the dataset. 

<b>aHash results</b>

 | Base Image Name         | Comparison Image Name              | Similarity Score |
 | ---------------------   | ---------------------              | ---:             
 | jennifer_aniston.jpeg   | jennifer_aniston_earrings.jpeg     | 0                
 | jennifer_aniston.jpeg   | natalie_portman_earrings.jpeg      | 24
 | jennifer_aniston.jpeg   | jennifer_aniston_earrings_03.jpeg  | 25
 | jennifer_aniston.jpeg   | poppy_delevingne_earrings.jpeg     | 27
 | jennifer_aniston.jpeg   | taylor_swift_earrings.jpeg         | 27
 | jennifer_aniston.jpeg   | hilary_swank_earrings.jpeg         | 28
 | jennifer_aniston.jpeg   | jennifer_aniston_earrings_02.jpeg  | 31 
 | jennifer_aniston.jpeg   | maggie_gyllenhaal_earrings.jpg     | 31
 | jennifer_aniston.jpeg   | nicole_kidman_earrings.jpeg        | 34
 | jennifer_aniston.jpeg   | julia_roberts_earrings.jpeg        | 34
 | jennifer_aniston.jpeg   | jennifer_garner_earrings.jpeg      | 35 
 | jennifer_aniston.jpeg   | elizabeth hurley_earrings.jpeg     | 41

The <i>average hash</i> algorithm was able to correctly classify 3 of the 6 variations of the Jennifer Aniston comparison image  within the modified dataset to the base image of Jennifer Aniston.  All the <i>Hamming distance</i> values for these modified images were in a range between 2 and 5, which are within the threshold range for potentially similar images. The <i>average hash</i> algorithm was not able to identify a mirror image of the base image within the modified dataset.

<p align="center">
  <img src="https://github.com/johnbumgarner/facial_similarities/blob/master/graphic/ahash_histogram_variations.png", width="600" height="600"/>
</p>

#### pHash algorithm

The core difference between the <i>average hash</i> algorithm and the <i>perceptive hash</i> algorithm is how the latter handles either gamma correction or color histogram modifications applied to an image.  The <i>average hash</i> algorithm will generate false-misses when slight color variations have been applied to the 
a comparison image. The <i>perceptive hash</i> algorithm handles these variations by using <i>discrete cosine transform</i>(DCT), which expresses a finite sequence of data points in terms of a sum of cosine functions oscillating at different frequencies.

The <i>perceptive hash</i> algorithm is designed to scale the input image down to 32×32 pixels and covert this smaller image to grayscale. Next the algorithm uses DCT to separates the image into a collection of frequencies and scalars. After this is done the algorithm extracts the top-left 8x8, which represent the lowest frequencies in the image.   The 64 bits of this 8x8 will be set to a binary value of 0 or 1 depending on whether the value is above or below the average value. The resulting hash value will not dramatically change even if a comparison image has had gamma or color histogram adjustments.

A <i>Hamming distance</i> value of 0 means that two images are identical, whereas a distance of 10 or less indicates that two images are potentially similar and a value greater than 10 suggests that the images are most likely different. 

The basic usage of the <i>perceptive hash</i> algorithm within <i>ImageHash</i> is:

```python
hash0 = imagehash.phash(Image.open(base_image))
hash1 = imagehash.phash(Image.open(comparison_image))
computational_score = (hash0 - hash1)
```

The <i>perceptive hash</i> algorithm correctly matched the Jennifer Aniston base image to the same Jennifer Aniston comparison image within the dataset.  The algorithm did not find any similarities between the Jennifer Aniston base image and the other 2 Jennifer Aniston comparison images within the dataset. 

<b>pHash results</b>

| Base Image Name         | Comparison Image Name               | Similarity Score |
| ---------------------   | ---------------------               | ---:  
| jennifer_aniston.jpeg   | jennifer_aniston_earrings.jpeg      | 0
| jennifer_aniston.jpeg   | elizabeth hurley_earrings.jpeg      | 24
| jennifer_aniston.jpeg   | poppy_delevingne_earrings.jpeg      | 26
| jennifer_aniston.jpeg   | hilary_swank_earrings.jpeg          | 26
| jennifer_aniston.jpeg   | natalie_portman_earrings.jpeg       | 26
| jennifer_aniston.jpeg   | jennifer_aniston_earrings_02.jpeg   | 28
| jennifer_aniston.jpeg   | maggie_gyllenhaal_earrings.jpg      | 28
| jennifer_aniston.jpeg   | julia_roberts_earrings.jpeg         | 28
| jennifer_aniston.jpeg   | jennifer_aniston_earrings_03.jpeg   | 30
| jennifer_aniston.jpeg   | jennifer_garner_earrings.jpeg       | 30
| jennifer_aniston.jpeg   | taylor_swift_earrings.jpeg          | 34
| jennifer_aniston.jpeg   | nicole_kidman_earrings.jpeg         | 38

The <i>discrete cosine transform</i> approached was able to correctly classify the 5 of the 6 variations of the Jennifer Aniston comparison image  within the modified dataset to the base image of Jennifer Aniston.  All the <i>Hamming distance</i> values for these modified images were in a range between 2 and 8, which are within the threshold range for potentially similar images. The <i>perceptive hash</i> algorithm was not able to identify a mirror image of the base image within the modified dataset.

<p align="center">
  <img src="https://github.com/johnbumgarner/facial_similarities/blob/master/graphic/phash_histogram_variations.png", width="700" height="700"/>
</p>


#### dHash algorithm

The <i>difference hash</i> algorithm is nearly identical to the <i>average hash</i> algorithm.  dhash is designed to tracks gradients, while aHash focuses on average values and pHash evaluates frequency patterns. dHash scales the input image down to an odd aspect ratio of 9x8. This aspect ratio has 72 pixels, which is slightly more then aHash's 64 pixels.  dHash coverts this smaller image to grayscale, which changes the hash from 72 pixels (72 red, 72 green, and 72 blue) to 72 total colors.  

After this conversion the dHash algorithm will measure the differences between adjacent pixels, thus identifying the relative gradient direction of each pixel on a row to row basis.  After all this computation the 8 rows of 8 differences becomes 64 bits.  Each of these bits is simply set based on whether the left pixel is brighter than the right pixel.  These measurements will match any similar image regardless of its aspect ratio prior to dHash shrinking the image.  

A <i>Hamming distance</i> value of 0 means that two images are identical, whereas a distance of 10 or less indicates that two images are potentially similar and a value greater than 10 suggests that the images are most likely different. 

The basic usage of the <i>difference hash</i> algorithm within <i>ImageHash</i> is:

```python
hash0 = imagehash.dhash(Image.open(base_image))
hash1 = imagehash.dhash(Image.open(comparison_image))
computational_score = (hash0 - hash1)
```

The <i>difference hash</i> algorithm correctly matched the Jennifer Aniston base image to the same Jennifer Aniston comparison image within the dataset.  The algorithm did not find any similarities between the Jennifer Aniston base image and the other 2 Jennifer Aniston comparison images within the dataset. 

<b>dHash results</b>

| Base Image Name         | Comparison Image Name               | Similarity Score |
| ---------------------   | ---------------------               | ---:  
| jennifer_aniston.jpeg   |  jennifer_aniston_earrings.jpeg     | 0
| jennifer_aniston.jpeg   |  hilary_swank_earrings.jpeg         | 18
| jennifer_aniston.jpeg   |  poppy_delevingne_earrings.jpeg     | 21
| jennifer_aniston.jpeg   |  jennifer_aniston_earrings_02.jpeg  | 22
| jennifer_aniston.jpeg   |  taylor_swift_earrings.jpeg         | 25
| jennifer_aniston.jpeg   |  julia_roberts_earrings.jpeg        | 26
| jennifer_aniston.jpeg   |  natalie_portman_earrings.jpeg      | 27
| jennifer_aniston.jpeg   |  elizabeth hurley_earrings.jpeg     | 28
| jennifer_aniston.jpeg   |  maggie_gyllenhaal_earrings.jpg     | 28
| jennifer_aniston.jpeg   |  nicole_kidman_earrings.jpeg        | 32
| jennifer_aniston.jpeg   |  jennifer_garner_earrings.jpeg      | 32
| jennifer_aniston.jpeg   |  jennifer_aniston_earrings_03.jpeg  | 35

#### wavelet algorithm

The <i>wavelet hash</i> algorithm is similar to the <i>perceptive hash</i> algorithm, because it operates within the frequency domain.  The main difference is that the <i>wavelet hash</i> algorithm uses <i>discrete wavelet transform</i>(DWT), instead of <i>discrete cosine transform</i> like the <i>perceptive hash</i> algorithm does. In numerical <i>analysis</i> and <i>functional analysis</i>, a <i>discrete wavelet transform</i> is any wavelet transform for which the wavelets are discretely sampled. The <i>wavelet hash</i> algorithm used the <i>Haar wavelet</i>, which is a sequence of rescaled "square-shaped" functions which together form a wavelet family.

The basic usage of the <i>wavelet hash</i> algorithm within <i>ImageHash</i> is:

```python
hash0 = imagehash.whash(Image.open(base_image))
hash1 = imagehash.whash(Image.open(comparison_image))
computational_score = (hash0 - hash1)
```
The <i>wavelet hash</i> algorithm has a mode parameter, which allows the wavelet family to be changed.  

```python
imagehash.whash(Image.open(image, hash_size = 8, image_scale = None, mode = 'haar', remove_max_haar_ll = True))
```
These wavelet families can be changed by installing the Python module <i>pywavelets</i>.

```python
import pywt
pywt.families()
['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']

# db3 is a family member of db
w = pywt.Wavelet('db3')

hash0 = imagehash.whash(Image.open(base_image) mode=w)
```

The <i>wavelet hash</i> algorithm correctly matched the Jennifer Aniston base image to the same Jennifer Aniston comparison image within the dataset.  If the computational score threshold was set to less than 15, then the other Jennifer Aniston's images within the dataset were not considered similar images.   

<b>wavelet results</b>

| Base Image Name         | Comparison Image Name               | Similarity Score |
| ---------------------   | ---------------------               | ---:  
| jennifer_aniston.jpeg   | jennifer_aniston_earrings.jpeg      | 0
| jennifer_aniston.jpeg   | jennifer_aniston_earrings_03.jpeg   | 24
| jennifer_aniston.jpeg   | poppy_delevingne_earrings.jpeg      | 26
| jennifer_aniston.jpeg   | taylor_swift_earrings.jpeg          | 26
| jennifer_aniston.jpeg   | natalie_portman_earrings.jpeg       | 26
| jennifer_aniston.jpeg   | hilary_swank_earrings.jpeg          | 28
| jennifer_aniston.jpeg   | jennifer_aniston_earrings_02.jpeg   | 30
| jennifer_aniston.jpeg   | nicole_kidman_earrings.jpeg         | 34
| jennifer_aniston.jpeg   | julia_roberts_earrings.jpeg         | 34
| jennifer_aniston.jpeg   | jennifer_garner_earrings.jpeg       | 36
| jennifer_aniston.jpeg   | maggie_gyllenhaal_earrings.jpg      | 38
| jennifer_aniston.jpeg   | elizabeth hurley_earrings.jpeg      | 40

The <i>discrete wavelet transform</i> approached was able to correctly classify the 6 variations of the Jennifer Aniston comparison image within the modified dataset to the base image of Jennifer Aniston.  All the computational values for these modified images were in a range between 2 and 12, which were all within the threshold range for potentially similar images.  The <i>wavelet hash</i> algorithm was also able to identify a mirror image of the base image, but the computational score was 16, which was slightly outside the threshold of 15 or less.

<p align="center">
  <img src="https://github.com/johnbumgarner/facial_similarities/blob/master/graphic/whash_histogram_variations.png", width="700" height="700"/>
</p>

</p>




.....
All four hashing methods were able to accurately identify the two images of Jennifer Aniston.  All the hashing methods produce similarity scores.  The threshold of these scores are adjustable.  
....



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



