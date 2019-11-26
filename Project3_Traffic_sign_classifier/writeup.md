# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

---
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://view5f1639b6.udacity-student-workspaces.com/notebooks/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

###### Image Visulaization
 - Here All German Traffic Sign Shown Once with their Label
 - Total count of same raffic sign image is available with the label 
   - For example : No Entry = 360.0 means No Entry signal Image is available 360 times in training data 
   

![Image Sample](https://view5f1639b6.udacity-student-workspaces.com/files/CarND-Traffic-Sign-Classifier-Project/examples/visulization.JPG)

---

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

 - As a 1st step, I decided to convert the images to grayscale because any operation on image is easy if it is converted to gray and model also learn fast. so the training speed will increase

- Here is an example of a traffic sign image before and after grayscaling.

![alt text](https://view5f1639b6.udacity-student-workspaces.com/files/CarND-Traffic-Sign-Classifier-Project/examples/grayscale.jpg)


 - after that i equalize the histogram for the image.

 - As a last step, I normalized the image data to convert it in range of -0.5 to 0.5 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| Tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|    
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 10x10x16 	|
| Tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|  
| Fully connected-1		| input 400, output 120							|
| Tanh					|												|
| Fully connected-1		| input 120, output 84							|
| Tanh					|												|
| Fully connected-1		| input 84, output 43							|
| Tanh					|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

- Here I Choose Learning rate = 0.0005 as i low learning rate can increase the model accuracy. 
- Optimizer which i choose is Adam Optimizer to train the model
- I choose 50 Epoch and Batchsize = 90 to train the model

###### Experment with Different Hyperparameters  
- I trained model with different values of ephoc and batch_size
    1.epoch = 100 and batch_size = 90 : Accuracy = Approx 94% but overfitting was there
    2.epoch = 50 and batch_size = 50 : Accuracy = Approx 93%.   
- Previously i used the RELU activation function but i got accuracy 93.2% with 50 epochs and 90 batch_size. but then i replaced Relu activation function with Tanh and i got 95.3% validation accuracy with same epoch and same batch_size 
  
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100 %
* validation set accuracy of 95.3 % 
* test set accuracy of 92.3 %

My Approch to get this result is discribe below

###### 4.1. Image Preprocesing 

 - i initially apply two pre-processing steps to our images:
 - Grayscale : i convert 3 channel image to a single grayscale image (we do the same thing in project 1 — Lane Line Detection
 - i convrted Image to the gray scale using cv2.cvtColor function. it will make processing speed on image to be faster
 - I applied Histogram equalizetion method using equilizeHist function.
 - in the last step i normalize the image 
 - I obtained the best experimental performance with histogram equalization and normalizing the image values from -0.5 to 0.5. 
 
###### 4.2. Reshape Images
 - After normalizing Image i converted Image into numpy array and reshape the array.
 
###### 4.3. Optimizer selection 
 - I used Adam Optimizer because it has benefits of Adaptive Gradient Algorithm and Root Mean Square Propagation. 

###### 4.4. Number and type of layers
 - To Train data i used Lenet Model from the last exercise with some changes.
    - my changes : 
        - output of Fully connected layer 3 is changed from 10 to 43
        - RELU activation function replaced with Tanh activation functiom
    
 - Detail information of my Layer architecture is available in below table.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| Tanh					| Activation Function  							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|    
| Convolution 3x3	    | 1x1 stride, Valid padding, outputs 10x10x16 	|
| Tanh					| Activation Function  							|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|  
| Fully connected-1		| input 400, output 120							|
| Tanh					| Activation Function  							|
| Fully connected-1		| input 120, output 84							|
| Tanh					| Activation Function  							|	 											
| Fully connected-1		| input 84, output 43							|
| Tanh					| Activation Function  							|		

 - First i did convolution of gray image(32x32x1) with 5x5 mask with stride of 1 then i applied tanh activation function.
 - after that i did sampling task by using max_pooling method. here i used stride of 2 in max pooling operation with valid padding. after this step i used tanh activation function
 - again i did convolution of the output of the max_pooling with 5x5 mask and the stride which i used for this operation is 1.after that again i used tanh activation function
 - I did sampling operation on the output of the second convoluation layer.
 - then i Flatten this sampling output 
 - i used 3 fully connected layer to get final output.
 - 1st fully connected layer have input = 400 and output of 120
 - 2nd fully connected layer have input = 120 and output of 84
 - 3rd fully connected layer have input = 84 and output of 43. here at the final stage we have 43 output of the 43 class.
 - each output is containing the probability of the input image with respect to the 43 labels.
   ex : suppose input is "STOP" sign so the output which is representing STOP sign having the higher probability compare to others.
 
 
###### 6. Model training
 - I decided on a simple learning rate of 0.0005 with 50 epochs, using a batch size of 90. The weights were initialized with TensorFlow's truncated_normal method, with a mean of 0 and a standard deviation of 0.1. The loss was calculated by applying a softmax cross entropy function, comparing the predicted classes with the validation set. This is then optimized with the tf.train.AdamOptimizer, which uses Kingma and Ba's Adam algorithm for first-order gradient-based optimization of randomized objective functions. Adam enables the model to use a large step size and move quickly to convergence without a lot of fine tuning.

###### 6. Model's assessment
 - Below is the accuracy detail which i got by using this model
 ![accuracy](https://view5f1639b6.udacity-student-workspaces.com/files/CarND-Traffic-Sign-Classifier-Project/result_images/accuracy.JPG)

My final model results were:
* training set accuracy of 100 %
* validation set accuracy of 95.3 % 
* test set accuracy of 92.3 %

 - Check the below graph of training and validation accuracy vs epochs
 ![plot](https://view5f1639b6.udacity-student-workspaces.com/files/CarND-Traffic-Sign-Classifier-Project/examples/plot.JPG)

---

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Using Google image search, I tried to select a few good representations of traffic signs to run through my model
nearly every image I found was shot with good lighting conditions and nearly always perfectly head on. Here are the best candidates I could come up with in my search ,along with the preprocessed versions I actually ran through the model

Here are the results of the prediction:
![Online_image](https://view5f1639b6.udacity-student-workspaces.com/files/CarND-Traffic-Sign-Classifier-Project/result_images/online%20image.jpg)

###### Steps to Check the accuracy of Online Downloaded Images
 - First we need to preprosessed the Images (converting to gray,histogram equilization and normalization) 
 - then we need to evalute the prediction of label of each online images based on the model which we saved previously. 
 - i found the total accuracy of 100% on the online downloaded Images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).



| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)      					| 
| Speed limit (70km/h)	| Speed limit (70km/h)							|
| No passing			| No passing									|
| Roundabout mandatory  | Roundabout mandatory	        				|
| Stop       			| Stop              							|
| Turn right ahead   	| Turn right ahead   							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.80         			| Speed limit (30km/h)   						| 
| 0.72     				| Speed limit (70km/h) 							|
| 0.99					| No passing									|
| 0.99	      			| Roundabout mandatory					 		|
| 0.99				    | Stop      							        |
| 0.95                  | Turn right ahead                              |

 - To see the first 5 softmax probebilities on the online image please refer the  HTML or .ipynb file.
