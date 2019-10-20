# Finding Lane Lines on the Road
---
The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---
### Lane Detection Pipeline 

#### Step1 : 
- In First step conversion of RGB image to GrayScale Image is required to process this image.
- We used ***grayscale*** function to convert image into gray scale
    -    Functionality : Convert Image to GrayScale 
    - input : RGB Image
    - Output : Gray Image
 - this grayscale function is using opencv function cv2.cvtColor.
 - Refer Output_Image : CarND-LaneLines-P1/Stepwise_Output_Images/Step1_gray_image.png
 
#### Step2 : 
- In Second step we need to Convert our Gray Image in Blur Image
- Blur Image is Smooth image so it will be useful in edge detection 
- here we used ***gaussian_blur*** to apply gaussian bluring effect on our Gray Image
    - Functionality : Appluy GussianBlur to make image smooth 
    - input :  Gray_Image, Kernal_size
    - Output : Blur_Image
- here ***kernal size*** is size of mask which will do convolution with image and give blur image
- gaussian_blur function is using opencv function cv2.GaussianBlur.
- Refer Output_Image : CarND-LaneLines-P1/Stepwise_Output_Images/Step2_Blur_image.png

#### Step3 : 
- After Appling Blur effect to Gray Image We need to detect edge from this Image
- For edge detection we used canny edge detection algorithm
- we are using ***canny*** function to detect edges from the blur image
    - Functionality : Appluy Canny Edge Detection to indentify Edges 
    - input : Blur_Image,Low_Thresold, High_Tresold
    - Output : Edge_Detected_Image
- canny function in pipeline is using opencv function cv2.canny to detect edges from the image
- Referance : [Canny Edge Detector](https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html)
- Refer Output_Image : CarND-LaneLines-P1/Stepwise_Output_Images/Step3_canny_image.png

#### Step4 : 
- From the Edge detected image mask the unuseful area and select your ROI 
- We Need to create our polygon (ROI) first then need to mask the part out side of the ROI form the edge detected Image
- We use ***region_of_interest*** function to do the same
   - Functionality : Mask Image with according to polygon 
   - input : Image,Polygon
   - Output : Edge detected Image with ROI
- Refer Output_Image : CarND-LaneLines-P1/Stepwise_Output_Images/Step4_ROI_Image.png

#### Step5 : 
- So now we have Image with Edges in ROI so we need to convert this Image space in Hough Space to find out the Lines 
- We used Function ***hough_lines*** Function to get Lines and Draw the Line on the Images
    - Functionality : Perform Hough Transformation and draw Lines
    - In : Image,rho,theta,thresold,min_line_distance,max_line_gap
    - out : Hough lines in blank Image
- hough_lines function is using opencv function cv2.HoughLinesP to detect the lines
- After getting this line ***draw_line*** function is used to draw this lines on Images
- Referance : [Hough Transorm](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html)

- ##### Step5.1 (Draw_Lines)
  -  using HoughLinesP we will get the lines with segments.
  -  we can get x1,x2,y1,y2 points from this line
  -  Joining this Line segments is devided in two part.
     ###### Step 5.1.1
     - calculate the slop and intercept of each line. 
     - seperate slop and intercept of right side and left side lines. 
     - average the slop and intercept of both side lines.
     ###### Step 5.1.2
     - define the ymax and ymin for connecting the lines
     - from the slop and intercept average calculate the x1 and x2 points for both the side
     - draw the line using cv2.line function.
- Refer Output_Image : CarND-LaneLines-P1/Stepwise_Output_Images/Step5_Line_image.png

#### Step6 : 
- after Getting the Line on the blank image we need to wight this line image with the original image.
- ***weighted_img*** is used to do this task 
  - Functionality :  Draw lines on edge image
  - In : HoughImage,Main_Image
  - out : Image with Lines
- it is using the opencv function cv2.addWeighted
- Refer Output_Image : CarND-LaneLines-P1/Stepwise_Output_Images/Step6_wighted_image.png

---

### Identify potential shortcomings with your current pipeline
- here line segment is connected so this algorithm will not work on curved lanes
- in Video line is littlebit shaking. 
- lane line which is very light color can not be detected

---

### Suggest possible improvements to your pipeline
- use proper detection method to detect lane line on light color lane.
- lane should be detected properly on curved road by updating draw_line function.







