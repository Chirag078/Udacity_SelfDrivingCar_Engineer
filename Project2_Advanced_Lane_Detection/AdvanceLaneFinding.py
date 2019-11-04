#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import glob
import pickle

# ### SideBySide 
# 
# - This is the Debug Function
# - take 2 images as input and display them side by side
# 

# In[ ]:


def SideBySide(img1,img2):
    fig, axs = plt.subplots(1, 2, figsize=(24, 100))
    
    count = 0
    for ax in axs:
        if count == 0:
            ax.imshow(img1)
            count+=1
        else:
            ax.imshow(img2)
            count=0


# ### Gradiant Thresholding
# 
#    ###### - AbsSobelThresold  :  
#    - it will perform sobel hresold 
#    ###### - MeanSobelThresold
#    
#    

# In[ ]:


def MeanSobelThersold(img,threshold=(0,255)):
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(gray_img,cv2.CV_64F,1,0)
    
    abs_img = np.sqrt(sobelx**2 + sobely**2)  
    scaler_img = np.uint8(255*(abs_img/np.max(abs_img)))
    grad_binary = np.zeros_like(abs_img)
    grad_binary[(scaler_img > threshold[0]) & (scaler_img < threshold[1])] = 1
    return grad_binary


# In[ ]:


def AbsSobelThresold(img,axis,threshold=(0,255)):
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if axis == 'X':
        sobel_img = cv2.Sobel(gray_img,cv2.CV_64F,1,0)
    if axis == 'Y':
        sobel_img = cv2.Sobel(gray_img,cv2.CV_64F,0,1)
    
    abs_img = np.absolute(sobel_img)
    scaler_img = np.uint8(255*(abs_img/(np.max(abs_img))))
    
    grad_binary = np.zeros_like(abs_img)
    grad_binary[(scaler_img > threshold[0]) & (scaler_img < threshold[1])] = 1
    return grad_binary


# In[ ]:


def DirThresold(img,thresold = (0,np.pi/2)):
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobel_imgx = cv2.Sobel(gray_img,cv2.CV_64F,1,0)
    sobel_imgy = cv2.Sobel(gray_img,cv2.CV_64F,0,1)
    
    abs_imgx = np.absolute(sobel_imgx)
    abs_imgy = np.absolute(sobel_imgy)
    
    direction = np.arctan2(abs_imgy,abs_imgx)
    direction = np.absolute(direction)

    d_binary = np.zeros_like(direction)
    d_binary[(direction >= thresold[0]) & (direction <= thresold[1])] = 1
    return d_binary


# In[ ]:


def ColorThresold(img,threshold=(0,255)):
    hls_img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    s_chennel = hls_img[:,:,2]
    s_binary = np.zeros_like(s_chennel)
    s_binary[(s_chennel > 150) & (s_chennel < 200)] = 1
    
    r_chennel = img[:,:,0]
    r_binary = np.zeros_like(r_chennel)
    r_binary[(r_chennel > threshold[0]) & (r_chennel <= threshold[1])] = 1
    
    mixedColor = np.zeros_like(r_binary)
    mixedColor[(s_binary == 1) | (r_binary == 1) ] = 1
    return mixedColor


# In[ ]:


def GradiantThresold(img):
    absSobel = AbsSobelThresold(img,"X",(30,100))
    meanSobel = MeanSobelThersold(img,(20,200))
    dirSobel = DirThresold(img,(0.7,1.3))
    colorThrsold = ColorThresold(img,threshold = (230,255)) 
    combined = np.zeros_like(absSobel)
    
    combined[((absSobel == 1) & (meanSobel == 1))| (dirSobel == 1) & (colorThrsold == 1) ] = 1
    return combined


# In[ ]:


def PerspectiveT(img):
    ImgSize = (img.shape[1], img.shape[0])
    src = np.float32(
        [[200, 720],
         [1180, 720],
         [600, 470],           # y = 470
         [750, 470]])
    
    dst = np.float32(
        [[300, 720],
        [980, 720],
        [300, 0],
        [980, 0]])
    

    
    
    a = [200,720]
    b = [1180,720]
    c = [550,470]
    d = [770,470]
    pts = np.array([a,b,d,c])
    
    blackimg = np.zeros_like(img)
    
    mask=cv2.drawContours(blackimg,[pts],0,(255,255,255),-1)
    new_test_img = cv2.bitwise_and(img,mask)

    
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, m, ImgSize, flags=cv2.INTER_LINEAR)
    
    M_inv= cv2.getPerspectiveTransform(dst, src)
    return warped,M_inv,new_test_img


# In[ ]:


def Visulization(out_black_img,leftx,lefty,rightx,righty,left_fitx,right_fitx,ploty):
    out_black_img[lefty, leftx] = [255, 0, 0]
    out_black_img[righty, rightx] = [0, 255, 0]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='orange')
    plt.plot(right_fitx, ploty, color='orange')
    
    return out_black_img


# In[ ]:


def SlidingWindow(binary_warped):

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # To Display Output
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # base left and right
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # HyperParameter
    nwindow = 10
    minpix = 50
    margin = 50
    height = np.int(binary_warped.shape[0]//nwindow)
    
    # NonZero in X & Y from Binary Image
    nonzero = binary_warped.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindow):
        win_y_low = binary_warped.shape[0] - (window+1)*height
        win_y_high = binary_warped.shape[0] - (window)*height
        
        win_leftx_low = leftx_current - margin
        win_leftx_high = leftx_current + margin
        win_rightx_low = rightx_current - margin
        win_rightx_high = rightx_current + margin
        
        cv2.rectangle(out_img,(win_leftx_low,win_y_low),
        (win_leftx_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_rightx_low,win_y_low),
        (win_rightx_high,win_y_high),(255,0,0), 5) 
    
        good_leftx_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_leftx_low) &  (nonzerox < win_leftx_high)).nonzero()[0] 
        good_rightx_inds = ((nonzerox >= win_rightx_low) & (nonzerox < win_rightx_high)
                          & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        left_lane_inds.append(good_leftx_inds)
        right_lane_inds.append(good_rightx_inds)
        
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_leftx_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_leftx_inds]))
        if len(good_rightx_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_rightx_inds]))
            
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except TypeError:
        pass
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    
    left_fit_coef,right_fit_coef,leftx_fit,rightx_fit,ploty = fitPoly(binary_warped,leftx,lefty,rightx,righty)
    visulize_img = None
    #visulize_img = Visulization(out_img,leftx,lefty,rightx,righty,leftx_fit,rightx_fit,ploty)
    
    return visulize_img,left_fit_coef,right_fit_coef,left_lane_inds,right_lane_inds,ploty
    


# In[ ]:


def fitPoly(binary_warped,leftx,lefty,rightx,righty):
    
    img_shape = (binary_warped.shape[1],binary_warped.shape[0])
    
    left_fit_coef = np.polyfit(lefty,leftx,2)
    right_fit_coef = np.polyfit(righty,rightx,2)
    
    ploty = np.linspace(0,img_shape[1]-1,img_shape[1])
    
    leftx_fit = left_fit_coef[0]*ploty**2 + left_fit_coef[1]*ploty + left_fit_coef[2]
    rightx_fit = right_fit_coef[0]*ploty**2 + right_fit_coef[1]*ploty + right_fit_coef[2]
    
    return left_fit_coef,right_fit_coef,leftx_fit,rightx_fit,ploty
    


# In[ ]:


def NearSearch(binary_warped,left_fit_coef,right_fit_coef,ploty):
    margin = 20
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    nonzero = binary_warped.nonzero()
    nonzerox = nonzero[1]
    nonzeroy = nonzero[0]
    
    left_lane_inds = ((nonzerox > (left_fit_coef[0]*(nonzeroy**2) + left_fit_coef[1]*(nonzeroy) + left_fit_coef[2] - margin)) &
                     (nonzerox < (left_fit_coef[0]*(nonzeroy**2) + left_fit_coef[1]*(nonzeroy) + left_fit_coef[2] + margin)))
                      
    right_lane_inds = ((nonzerox > (right_fit_coef[0]*(nonzeroy**2) + right_fit_coef[1]*(nonzeroy) + right_fit_coef[2] - margin)) &
                     (nonzerox < (right_fit_coef[0]*(nonzeroy**2) + right_fit_coef[1]*(nonzeroy) + right_fit_coef[2] + margin)))
                      
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
     
    left_fit_coef,right_fit_coef,leftx_fit,rightx_fit,ploty = fitPoly(binary_warped,leftx,lefty,rightx,righty)
    result = DrawNearArea(out_img,margin,nonzerox,nonzeroy,left_lane_inds,right_lane_inds,leftx_fit,rightx_fit,ploty)
    #out_black_img = Visulization(out_img,leftx,lefty,rightx,righty,leftx_fit,rightx_fit,ploty)
    return result,leftx_fit,rightx_fit


# In[ ]:


def DrawNearArea(out_img,margin,nonzerox,nonzeroy,left_lane_inds,right_lane_inds,left_fitx,right_fitx,ploty):
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    
    return result


# In[ ]:


def CalibrateCamera():
    nX = 9
    nY = 6
    
    objPoints = []
    imgPoints = []
    
    objp = np.zeros((nX*nY,3), np.float32)               # create nX*nY size matrix having 0
    objp[:,:2] = np.mgrid[0:nX,0:nY].T.reshape(-1,2)
    
    
    for calibImgName in os.listdir('./camera_cal'):
        calibImg = cv2.imread(os.path.join('./camera_cal',calibImgName))
        gray_calibImg = cv2.cvtColor(calibImg,cv2.COLOR_BGR2GRAY)
        
        # find chessBoard Corners
        ret,corners = cv2.findChessboardCorners(gray_calibImg,(nX,nY),None)
        
        if ret == True:
            objPoints.append(objp)
            imgPoints.append(corners)
            calibImg = cv2.drawChessboardCorners(calibImg,(nX,nY),corners,ret)
    
    
    ret,dist,mtx,rvecs,tvecs = cv2.calibrateCamera(objPoints,imgPoints,(calibImg.shape[1],calibImg.shape[0]),None,None)
    
    return dist,mtx


# In[ ]:


def UndistortImg(img,mtx,dist):
    undistorted_img = cv2.undistort(img,mtx,dist,None,mtx)
    return undistorted_img


# In[ ]:


def RadiusOfCurv(img,x_values):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # If no pixels were found return None
    y_points = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(y_points)

    # Fit new polynomials to x,y in world space
    print((y_points*ym_per_pix).shape,(x_values*xm_per_pix).shape)
    fit_cr = np.polyfit(y_points*ym_per_pix, x_values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad


# In[ ]:


def FinalImage(img,warped,M_inv,leftx_fit,rightx_fit,new_test_img):
    out_img = np.zeros_like(img)
    y_points = np.linspace(0, img.shape[0]-1, img.shape[0])

    left_line_window = np.array(np.transpose(np.vstack([leftx_fit, y_points])))

    right_line_window = np.array(np.flipud(np.transpose(np.vstack([rightx_fit, y_points]))))

    line_points = np.vstack((left_line_window, right_line_window))

    cv2.fillPoly(out_img, np.int_([line_points]), [0,255, 0])

    unwarped = cv2.warpPerspective(out_img, M_inv, (img.shape[1],img.shape[0]) , flags=cv2.INTER_LINEAR)
    print(img.shape)
    print(img.dtype)
    print(unwarped.shape)
    print(unwarped.dtype)
    result = cv2.addWeighted(img, 1, unwarped, 0.3,0)
    SideBySide(result,new_test_img)

    


# In[ ]:


def AdvancePipeline(img):
    img_size = (img.shape[1],img.shape[0])
    undistorted_img = UndistortImg(img,dist,mtx)
    
   
    edgeBinary = GradiantThresold(undistorted_img)
    #SideBySide(edgeBinary,undistorted_img)
    
    perspective_img,M_Inv,new_test_img = PerspectiveT(edgeBinary)

    visulize_img,left_fit_coef,right_fit_coef,left_lane_inds,right_lane_inds,ploty = SlidingWindow(perspective_img)
    #SideBySide(perspective_img,visulize_img)
    image,leftx_fit,rightx_fit = NearSearch(perspective_img,left_fit_coef,right_fit_coef,ploty)
    leftCurvRad = RadiusOfCurv(perspective_img,leftx_fit)
    rightCurvRad = RadiusOfCurv(perspective_img,rightx_fit)
    avgCurvRad = (leftCurvRad + rightCurvRad)/2
    #SideBySide(perspective_img,image)
    print(avgCurvRad)
    
    lane_center = (rightx_fit[719] + leftx_fit[719])/2
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    center_offset_pixels = abs(img_size[0]/2 - lane_center)
    center_offset_mtrs = xm_per_pix*center_offset_pixels
    offset_string = "Center offset: %.2f m" % center_offset_mtrs
    print(offset_string)
    FinalImage(img,perspective_img,M_Inv,leftx_fit,rightx_fit,new_test_img) # last arg remove


# In[ ]:


dist,mtx = CalibrateCamera()
for imgName in os.listdir('./test_images'):
    
    img = mpimg.imread(os.path.join('./test_images',imgName))
    AdvancePipeline(img)


# In[ ]:


video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image_Repeat) 
white_clip.write_videofile(AdvancePipeline, audio=False)


# In[ ]:





# In[ ]:





# In[ ]:




