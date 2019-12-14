# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## **Required Files & Quality of Code**

### **1. Submission includes all required files and can be used to run the simulator in autonomous mode**

My project includes the following files:
* behavior_cloning.py  = containing the script to create and train the model
* drive.py  = for driving the car in autonomous mode
* model.h5  = containing a trained convolution neural network 
* Behavioral_Cloning.mp4 = Video Recording of driving in autonomous mode
* writeup.md = summarizing the results

### **2. Submission includes functional code**
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### **3. Submission code is usable and readable**

The behavior_cloning.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

## **Model Architecture**

### **1. Model architecture** 
 - I have used nVIDEA Model architecutre as suggested by UDACITY with some changes.
 - The model includes ELU layers to introduce nonlinearity. 
 - Image has be cropped and normalize using Cropping2D and Lambda layer.
 - Layer Discription is available below.
  - Image normalization
  - Image Cropping
  - Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
  - Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
  - Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
  - Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
  - Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
  - Drop out (0.5)
  - Fully connected: neurons: 100, activation: ELU
  - Fully connected: neurons: 50, activation: ELU
  - Fully connected: neurons: 10, activation: ELU
  - Fully connected: neurons: 1 (output)
  
 - **nVIDEA Model Arch**


  ![alt text](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

  - The below is the **path of image for model structure output from the Keras** which gives more details on the shapes and the number of parameters.
 -  ./examples/Model_Arch

### **2. Attempts to reduce overfitting in the model**
 - I have used one Dropout layer after the last convolution layer with 50% of the input units to drop.
 - The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### **3. Model parameter tuning**
 - Below Parameters Used to train the model
  - Optimizer = Adam
  - loss = mean squared error "mse"

### **4. Fit generator parameters**
  - train_generator
  - samples_per_epoch= len(train_samples)
  - validation_data=validation_generator
  - nb_val_samples=len(validation_samples) 
  - nb_epoch=5
  -  verbose=1 

### **5. Appropriate training data**
 - I collected Data with 3 lap in center lane driving clockwise and 1 lap center lane driving anti clockwise. and for this data i have used 3 augmentation techniques -> zoom,horizontal flip and brightness change. (in this case i have used only camera images to train the model)

 - after that I Used Data which is provided by Udacity. (in this case i have used all 3 camera images to train the model and the flip version of the same images also)

- To augment the data sat, I also flipped images ,I have used total 6 Images from the same place of the track
  - three Camera Images 
    - Center camera Image
    - Left camera Image
    - right camera Image
  - three fliped version of camera images
    - center camera flip image
    - left camera flip image
    - right camera flip image
  
  - **Path for the Images** "./Examples/flip_original_images" 
    - center_images.png contain the original and flip version images from the center camera
    - left_images.png contain the original and flip version images from the left camera
    - right_images.png contain the original and flip version images from the right camera
    

 - I finally randomly shuffled the data set and put 20% of the data into a validation set. 

---
## References

NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
