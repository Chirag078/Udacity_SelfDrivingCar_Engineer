# Advanced Lane Detection
---
#### Goal : 
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
---
## [Rubric ](https://review.udacity.com/#!/rubrics/571/view) Points 

#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

Writeup / README
1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.
You're reading it!
---
## Camera Calibration
1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

##### Steps :-
- First i took the nX(number of corner in row) and nY(number of corner in column) to calibrate the camera
- then i  prepared "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0
- After that i am searching for the corners in all chess board images and if corner is found then i am appending objp into objectpoints and corners into imagepoints
- after this for conformation i am ploting the points again on images using corners which i got from previous step
- I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function.

![Calibration](https://github.com/Chirag078/Udacity_SelfDrivingCar_Engineer/blob/master/Project2_Advanced_Lane_Detection/examples/calib_image.JPG?raw=true)
---

## Advance Pipeline (Test Images)
##### 1 . Distortion removal on images
- Removed Distortion to the test image using camera calibration and distortion coefficients and  cv2.undistort() function.

![Undistorted Image](https://github.com/Chirag078/Udacity_SelfDrivingCar_Engineer/blob/master/Project2_Advanced_Lane_Detection/Undistorted_Op_Images/test3.jpg?raw=true)

##### 2 . Application of color and gradient thresholds to focus on lane lines 
- I used a combination of color and gradient thresholds to generate a binary image
    Gradient & Color Threshold :
    - Absolute Sobel in x direction 
    - R and S Channel as color Gradient

![Threshold Image](https://github.com/Chirag078/Udacity_SelfDrivingCar_Engineer/blob/master/Project2_Advanced_Lane_Detection/Threshold_Op_Images/test3.jpg?raw=true)

##### 3 . Production of a bird’s eye view image via perspective transform
 - I verified that my perspective transform was working as expected by drawing the src and dst points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

    src = np.float32(                                                    
        [[570, 470],
         [200, 720],
         [1110, 720],           
         [722, 470]])
    
    dst = np.float32(                                           
        [[320, 0],
        [320, 720],
        [960, 720],
        [960, 0]])

![Warped Image](https://github.com/Chirag078/Udacity_SelfDrivingCar_Engineer/blob/master/Project2_Advanced_Lane_Detection/Warped_Op_Images/test3.jpg?raw=true)
		
##### 4 . Fitting of second degree polynomials to identify left and right lines composing the lane
- Using Histogram i found the peak Points which is starting points of the lane
- then i used Sliding window concept on bird eye view image and applied second order polynomial

![Warped Image](https://view5639f7e7.udacity-student-workspaces.com/files/CarND-Advanced-Lane-Lines/Window_Op_Images/test3.jpg)

##### 5 . Computation of lane curvature and deviation from lane center
 - Calculated the curvature from the below equation
 
![R_Curve](https://github.com/Chirag078/Udacity_SelfDrivingCar_Engineer/blob/master/Project2_Advanced_Lane_Detection/examples/Redius.JPG?raw=true) 

##### 6. Warping and drawing of lane boundaries on image as well as lane curvature information
 - here it my out put image with Warping and drawing of lane boundaries on image as well as lane curvature information
 
![Final_Output](https://github.com/Chirag078/Udacity_SelfDrivingCar_Engineer/blob/master/Project2_Advanced_Lane_Detection/Final_Op_Images/test3.jpg?raw=true) 

---

## Advance Pipeline (Project Video)

[Lane Finding Project Video](https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/7cb63828-36aa-4cea-9239-700b5ea41f0b/concepts/0a96d23f-6c22-4053-a7f6-83e12ce5a6ec)

---

## Discussion


#### Issue Which i Faced is listed below.
 -  it is like to consider to which thresold need to take is difficult. because if i am selecting r and s color chennal then it is working properly but in binary image it is not looking properly. but i tried with different color chennal and got the best out put for "Project Video" using r and s chennal selection 
 
 - so now second issue is in sobel thresold value. here i used (50,150) as a thresold value so i am not able to load the challenge video but i changed this value and i succed to load that video and it that vido also i am getting some what clear output.
 
#### Improvment
 - to make this algorithm robust we need to use Different color chennal and threshold combination. and need to select the best one