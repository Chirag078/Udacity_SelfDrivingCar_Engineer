import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def ImgCalibrateCamera(nx,ny,DirPath):
    
    ImagesForCameraCal = os.listdir(DirPath)                                 
    
    '''
    obj_points = to store the objectpoints ex: (0,0,0),(1,0,0),(2,0,0),...,(9,6,0)
    img_points = to store the corners of all calib images 
    '''
    obj_points = []
    img_points = []
    
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    '''
    For each image 
     -> get the full image path
     -> read image in gray scale
     -> find the chessford corner by using cv2.findChessboardCorners
     -> if corner founded :
         -> append the objp to obj_point
         -> append the corner to img_point 
         -> (to test - Draw the corners on images by using cv2.drawChessboardCorners)
         -> calibrate the camera by using cv2.calibrateCamera
         -> return camera matrix and distortion coefficient
    '''
    for image in ImagesForCameraCal:                
        ImgPath = os.path.join(DirPath,image)                               
        GrayImg = cv2.imread(ImgPath,0)
        ImageSize = (GrayImg.shape[1],GrayImg.shape[0])                                             
        
        
        ret,corners = cv2.findChessboardCorners(GrayImg,(nx,ny),None)
        if ret == True:
            obj_points.append(objp)
            img_points.append(corners)
            
            '''
            # This Code is to verify that Points are drawn on Tested Images are correct or not
            Img = cv2.drawChessboardCorners(GrayImg,(nx,ny),corners,ret)
            plt.imshow(Img)
            plt.show()
            '''
    
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(obj_points,img_points,ImageSize,None,None)
    return mtx,dist
    
if __name__ == "__main__":
    mtx,dist = CalibrateCamera(9,6,"./camera_cal")
    SaveDir = {'mtx': mtx, 'dist': dist}
    with open('calibrate_camera.p', 'wb') as f:
        pickle.dump(SaveDir, f)

    # Undistort example calibration image
    img = mpimg.imread('camera_cal/calibration5.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.imshow(dst)
    plt.show()
    plt.savefig('examples/undistort_calibration.png')
    