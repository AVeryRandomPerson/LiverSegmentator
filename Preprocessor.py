import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

CROP_SOURCE = "C:/Users/acer/Desktop/TestSamples/BodyOnly/Mixed/"
CROP_SAVE = "C:/Users/acer/Desktop/TestSamples/Cropped36x36/Mixed/NoLiver/"

#Images should be in .jpg
def cropImagesInFolder(h1,h2,w1,w2,fol_path):
    for file in os.listdir(fol_path):
        if file.endswith(".jpg"):
            img = cv2.imread(fol_path + file)
            cv2.imwrite(CROP_SAVE + file, img[h1:h2,w1:w2])

cropImagesInFolder(190,226,320,356,CROP_SOURCE)

def getHistograms(fol_path):
    final_hist = np.zeros((1,256))
    for file in os.listdir(fol_path):
        if file.endswith(".jpg"):
            image = cv2.imread(fol_path+file)

            cur_hist = cv2.calcHist([image],[0],None,[256],[0,256])
            final_hist = np.add(final_hist,cur_hist)

    return final_hist



def thresholdImagesInFolder(fol_path,tresh_min,tresh_max):
    for file in os.listdir(fol_path):
        if file.endswith(".jpg"):
            image = cv2.imread(fol_path+file)
            ret,thresh_im = cv2.threshold(image,tresh_min,tresh_max,cv2.THRESH_TOZERO)
            cv2.imwrite(fol_path+"threshed/"+file,thresh_im)




class RangeFilter:

    # Given a folder directory,
    def sampleIntensity(self):
        print("Nothing here yet")
