# Traffic Signs Recognition

**In this Python project, we will build a deep neural network model that can classify traffic signs present in the image into different categories. With this model, we are able to read and understand traffic signs which are a very important task for all autonomous vehicles.
**

**Dataset: German Traffic Sign Dataset. **

** Data Set**

Dataset can be download here: 
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

Dataset Overview :

More than 40 classes

More than 50,000 images in total

Physical traffic sign instances are unique within the dataset
(i.e., each real-world traffic sign only occurs once)

## 1-) Loading and Visualise The Data Set

- `train.p`: The training set.
- `test.p`: The testing set.
- `valid.p`: The validation set.

I will use Python `pickle` to load the data.
Dataset Summary & Exploration
The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.


**Then, I used `matplotlib` plot sample images .**

<figure>
 <img src="./traffic-signs-data/README-img/first.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

** Then, I used `seaborn` to plot a histogram of the count of images in each unique class.
<figure>
 <img src="./traffic-signs-data/README-img/seaborn.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

**And finally, I will use `numpy` to plot a histogram of the count of images in each unique class.**
<figure>
 <img src="./traffic-signs-data/README-img/hist.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


## 2-) Data Preprocessing-Normalization,Reshape,Local Histogram Equalization

**Actions to be taken for data preprocessing:
We will be able to perform the most accurate analysis by minimizing unwanted situations with the data preprocessing method.
 (Shuffling, Grayscaling, Local Histogram Equalization, Normalization.)**

**Shuffling: We mix data to increase randomness and diversity.**

**Grayscaling: I will use OpenCV to convert training images to gray scale.
Using grayscale images instead of color improves ConvNet's accuracy.**


<figure>
 <img src="./traffic-signs-data/README-img/gray.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>
**Local Histogram Equalization: This technique emits the most common intensity values in the image and enhances low contrast images.
The application of this technique will be very helpful in our case because the data set at hand has real-world images and many have low contrast.
I will use 'skimage' to apply local histogram synchronization to training images.**


<figure>
 <img src="./traffic-signs-data/README-img/LOCAL.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

**Normalization: The method we call normalization is the assignment of variables to values between 0 and 1.**

<figure>
 <img src="./traffic-signs-data/README-img/Normalization.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

## 3-) Convolutional Neural Network





