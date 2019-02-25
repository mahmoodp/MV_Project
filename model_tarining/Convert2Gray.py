

import glob

import os

import cv2

# Replace mydir with the directory you want
mydir = '\\intra.tut.fi\home\mahmoodp\My Documents\Desktop\Pattern\Week4\Dataset'

def converimg2gray(folder):
      
    subdirectories = glob.glob(folder + "/*")
        
    # Loop over all folders
    for d in subdirectories:
        print(d)
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
                
        # Load all files
        for name in files:
            
            image = cv2.imread(name) 
            print(image.shape)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
            path = mydir + str(subdirectories[0])
            cv2.imwrite(os.path.join(path,'{}'.format(name)),gray_image) 
            



converimg2gray("Dataset")
