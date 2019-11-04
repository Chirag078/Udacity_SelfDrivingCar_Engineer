import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np 
import os

for images in os.listdir("./test_images"):
    imgpath = os.path.join("./test_images",images)
    img = mpimg.imread(imgpath)
    
    hsl = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    l = img[:,:,0]
    
    r_chennel = img[:,:,0]
    r_binary = np.zeros_like(r_chennel)
    r_binary[(r_chennel > 150) & (r_chennel <= 255)] = 1
    '''
    u = img[:,:,1]
    v = img[:,:,2]
   
    plt.imshow(l,cmap='gray')
    plt.show()
    plt.imshow(u,cmap='gray')
    plt.show()
     '''
   
    s_chennel = hsl[:,:,2]
    s_binary = np.zeros_like(s_chennel)
    s_binary[(s_chennel > 150) & (s_chennel <= 255)] = 1
    mixed = np.zeros_like(s_binary)
    mixed[(r_binary == 1) & (s_binary == 1)] = 1
    plt.imshow(mixed,cmap='gray')
    plt.show()
