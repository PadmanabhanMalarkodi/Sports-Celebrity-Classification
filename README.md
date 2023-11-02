# Sports-Celebrity-Image-Classification
Welcome to the Sports Celebrity Image Classification project! In this repository, we have developed a machine learning model that can accurately classify images of famous sports celebrities. 
Whether you're a sports enthusiast, a fan of a particular athlete, or just curious to see if your model can identify your favorite sports stars, this project has you covered.

## Introduction
* Image classification refers to grouping the images based on similar features. It is a supervised learning approach in which you are given a labeled dataset.
* Classifying cat and dog images is a common computer vision project, and you can approach it using various machine learning algorithms.
 
* Here's a step-by-step guide to help you get started on this project:

## 1. Set Up Your Environment:
* Install Python and necessary libraries, such as NumPy, OpenCV, Python Wavelet Transform Module, Scikit-learn.

## 2. Data Acquisition:
* Download the images of celebrities and store it in a seperate folders.

## 3. Data Exploration:
### Once you have your dataset, it's essential to understand its characteristics and structure:

**Load the Dataset** 
Use Python and libraries like OpenCV to load the dataset.

**Explore the Data**
Check the dataset size, the distribution of an image and any potential data imbalances. Visualization libraries like Matplotlib or Seaborn can help you create insightful graphs and plots.

## 4. Data Preprocessing:
### Prepare your data for training by performing the following tasks:

**Convert the colored image into grayscale image**
* Convert the colored image into grayscale image as hog function in scikit-image is designed to work with grayscale images by default.

**Resize the image**
* Resize all the images to a consistent size (e.g., 32x32pixels).

**Face and Eye Detection Using OpenCV**
  * *Step 1*: Import the OpenCV Package.
  * *Step 2*: Read the Image.
  * *Step 3*: Convert the Image to Grayscale.
  * *Step 4*: Load the Classifier for face detection.
  * *Step 5*: Perform the Face Detection.
  * *Step 6*: Draw a Bounding Box for head.
  * *Step 7*: Display the Image.
  * *Step 8*: Similarly, Load the Classifier to detect eyes.
  * *Step 9*: Perform eyes detection.
  * *Step 10*: Draw a bounding box for eyes.

**Crop the image**
* Crop the desired portion (region of interest) from each image.

**Store the image**
* Store the cropped image in a seperate folder.

**Delete the unwanted images**
* The processed image might still have some unwanted image, we are manually deleting them.

## 5. Feature Extraction (Optional):
* Apply wavelet tranform to extract feature from each of the image.
* We will be vertically stacking the wavelet image and cropped image.
* Images in cropped folder can be used for model training. We will use these raw images along with wavelet transformed images to train our classifier.

## 6. Model Selection:
* Decide on the machine learning algorithm you want to use. You have several options:
* Traditional machine learning algorithms such as Logistic Regression, Support Vector Machines (SVM), Random Forest, or k-Nearest Neighbors (k-NN).
* We will be choosing SVM initially to train our model.

## 7. Model Training:
* We will be splitting the train and test data separately.
* Then we will be train our model with the train data set.

## 8. Model Evaluation:
* Use the validation dataset to evaluate your model's performance.

## 9. Model Fine-Tuning (Optional):
* If the model's performance is not satisfactory, consider adjusting hyperparameters, adding more layers, or increasing training epochs.
* Use GrindsearchCV to find the best model along with the desired parameters.

## 10. Testing:
* After you're satisfied with your model's performance on the validation dataset, test it on the test dataset.
* Print the confusion matrix and classification report which will have F1 score, precision, reacall and model score that helps in understanding the model's performance.
