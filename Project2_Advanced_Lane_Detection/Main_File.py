from Calibrate_Camera  import ImgCalibrateCamera
import cv2
import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Undestort_PrespectiveT import ImgUndistort
from Threshold import ImgWithAllThresold
from Undestort_PrespectiveT import ImgPerspectiveT
from Sliding_Window import FitPlynomial
from Area_From_PreviosFrame import search_around_poly
from Radius_of_curv import measure_curv
import numpy as np

def Advance_Pipeline(Img):
    mtx,dist = ImgCalibrateCamera(9,6,"./camera_cal")

    UndisortedImg = ImgUndistort(Img)

    ThresoldImg = ImgWithAllThresold(UndisortedImg)
    plt.imshow(ThresoldImg)
    plt.show()
    '''
    PerspectiveTImg,M_inv = ImgPerspectiveT(ThresoldImg)
    
    
    PriviousLeftFit,PriviousRightFit,out_img,left_lane_inds,right_lane_inds = FitPlynomial(PerspectiveTImg)
    
    new_img,left_fitx,right_fitx = search_around_poly(PerspectiveTImg,PriviousLeftFit,PriviousRightFit)
    measure_curv(out_img.shape,left_fitx,right_fitx)
    #plt.imshow(out_img)
    #plt.show()
    
    
    y_points = np.linspace(0, out_img.shape[0]-1, out_img.shape[0])
    
    left_line_window = np.array(np.transpose(np.vstack([left_fitx, y_points])))
    
    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, y_points]))))
    
    line_points = np.vstack((left_line_window, right_line_window))
    
    cv2.fillPoly(out_img, np.int_([line_points]), [0,255, 0])
    
    unwarped = cv2.warpPerspective(out_img, M_inv, (out_img.shape[1],out_img.shape[0]) , flags=cv2.INTER_LINEAR)
    
    print(Img.dtype())
    print(unwarped.dtype())
    
    
    result = cv2.addWeighted(Img, 1, unwarped, 0.3, 0)
    '''
    
if __name__ == "__main__":
    
    for Img in os.listdir("./test_images"):
        ImgDir = os.path.join("./test_images",Img)
        Image = mpimg.imread(ImgDir)
        Advance_Pipeline(Image)
        
        