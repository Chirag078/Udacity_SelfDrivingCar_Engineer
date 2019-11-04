from Calibrate_Camera  import ImgCalibrateCamera
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from asn1crypto._errors import unwrap
from color.color import black

def ImgUndistort(img):
    with open('calibrate_camera.p', 'rb') as f:
        Parametes = pickle.load(f)
    mtx = Parametes['mtx']
    dist = Parametes['dist']
    
    UndistortedImage = cv2.undistort(img,mtx,dist,None,mtx)
    return UndistortedImage

def ImgPerspectiveT(img):

    ImgSize = (img.shape[1], img.shape[0])
    src = np.float32(
        [[200, 720],
         [1180, 720],
         [600, 470],
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
    plt.imshow(new_test_img)
    plt.show()    
    
    
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, m, ImgSize, flags=cv2.INTER_LINEAR)
    
    M_inv= cv2.getPerspectiveTransform(dst, src)
    plt.imshow(warped)
    plt.show()
    return warped,M_inv

if __name__ == "__main__":
    image = cv2.imread("./test_images/test1.jpg")
    ImgPerspectiveT(image)
    
    cv2.drawChessboardCorners()()
   
    
