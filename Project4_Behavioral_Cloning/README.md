# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

* behavior_cloning.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* writeup.md (Report)
* Behavioral_Cloning.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)
---

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

---

## **Training Strategy**

### **1. Solution Design Approach**

 - #### 1.1 - Data Collection for Training

  - My first step was to use a convolution neural network model similar to the nVIDEA model but because of less data (i have used only center camera images only) car was not able to stay on track but training and validation Loss was low.

   - After that i Used Udacity data with all 3 cameras and i also used flip augmentation method. (here we can collect data again from the simulator but because of GPU time constrain i used data provided by UDACITY)

 - #### 1.2 - Generated more data by using Data Generator 
   - To increase the training data i have used data generator method to generate image on the fly. 
   - Image genetrator is genetrating image by using below method.
    - **3 Original Image**
       - Used All 3 Original Camera Images and steering angle position.
       - for steering angle correction i used correction fector = 0.2.
       - here we need to correct the steering angle for the left and right cameras becasue log file contain steering angle value with center camera.
    - **3 fliped version of Original Image**
       - I have Fliped this 3 images(left,right and center) and use this 3 fliped images also for training and also corrected the steering angle for this images by multipling -1 to the original steering angle of without flip images. 

 - #### 1.3 - Training and Validation Dataset Split
   - I have splited this data into training and validation set, 
   - 80% data for the Training 
   - 20% data for the validation 

 - #### 1.4 - Model Parameter
   - I have used below paramters for training the data
   - Optimizer = 'adam'
   - Loss = 'mse'

 - #### 1.5 - fit_generator parameters 
  - After that i Trained the model using fit_generator. i have used below values as a argument for fit_genetator function.
  - train_generator
  - samples_per_epoch= len(train_samples)
  - validation_data=validation_generator
  - nb_val_samples=len(validation_samples) 
  - nb_epoch=5
  -  verbose=1 

 - #### 1.6 - Drive Car in Autonoms Mode
  - The final step was to run the simulator to see how well the car was driving around track one. There were a few spots (mostly ending of the track) where the vehicle fell off the track. then i checked images and driving_log file and i found like there is some images are missing which is available in driving log so i corrected this by adding the more images.

 - At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

## **2. Final Model Architecture**

 - I have used the modified nVIDEA Model to clone the driving behaviour

### **nVIDEA Model Arch**
 ![alt text](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

### **Model Layers Discription**
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
   
The below is the path of image for model structure output from the Keras which gives more details on the shapes and the number of parameters.
 -  ./examples/Model_Arch
### **Why Conv2D and Fully connected layer ?**
- As per the NVIDIA model, the convolution layers are meant to handle feature engineering and the fully connected layer for predicting the steering angle. However, as stated in the NVIDIA document, it is not clear where to draw such a clear distinction. Overall, the model is very functional to clone the given steering behavior.




## **3. Creation of the Training Set & Training Process**

- I first recorded Three laps on track one using center lane driving in clockwise and one lap in anti clockwise. 

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

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I Used 5 epochs to train the model. I used an adam optimizer so that manually training the learning rate wasn't necessary.

---


## Details About Files **drive.py** and **video.py**

### **`drive.py`**

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

### **Saving a video of the autonomous agent**

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

### **Why create a video**

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

---

## **Setup Information**
 - Python3 with all required modules.
 - you can run code in google colob or in jupyter notebook (make sure the file path ex.input image path)
 - based on Input file path may be you need to change code little bit