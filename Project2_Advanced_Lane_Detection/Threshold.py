import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
        
    abs_sobel = np.absolute(sobel)
    scaler_sobel = np.uint8(255*(abs_sobel/(np.max(abs_sobel))))
    
    grad_binary = np.zeros_like(abs_sobel)
    grad_binary[(scaler_sobel >= thresh[0]) & (scaler_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
        
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaler_sobel = np.uint8(255*(abs_sobel/(np.max(abs_sobel))))
    
    mag_binary = np.zeros_like(abs_sobel)
    mag_binary[(scaler_sobel >= mag_thresh[0]) & (scaler_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
        
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    scaler_sobel = np.arctan(abs_sobelx,abs_sobely)
    dir_binary = np.zeros_like(abs_sobelx)
    dir_binary[(scaler_sobel >= thresh[0]) & (scaler_sobel <= thresh[1])] = 1
    return dir_binary


def color_thresold(img, s_thresh=(170, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return s_binary

def ImgWithAllThresold(img):
    
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20,100))
    #grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20,100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30,200))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    s_binary = color_thresold(img)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
    return combined



if __name__ == "__main__":
    image = cv2.imread("./test_images/test1.jpg")
    ImgWithAllThresold(image)