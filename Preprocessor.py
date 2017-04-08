import logging
import os

import cv2
import numpy as np

from imutils import paths
#refactoring needed.


CROP_SOURCE = "C:/Users/acer/Desktop/TestSamples/BodyOnly/Mixed/"
CROP_SAVE = "C:/Users/acer/Desktop/TestSamples/Cropped36x36/Mixed/NoLiver/"

#Images should be in .jpg
def cropImagesInFolder(h1,h2,w1,w2,fol_path):
    for file in os.listdir(fol_path):
        if file.endswith(".jpg"):
            img = cv2.imread(fol_path + file)
            cv2.imwrite(CROP_SAVE + file, img[h1:h2,w1:w2])



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



def binaryAND(src1,src2,out):
    logging.basicConfig(filename="C:/Users/acer/Desktop/TestSamples/ML-Dataset/Bin-Results/bitAND_log.txt", level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    fol1_img = []
    fol2_img = []

    for file in os.listdir(src1):
        fol1_img.append(file)

    for file in os.listdir(src2):
        fol2_img.append(file)

    for i in range(0,len(fol2_img)):
        img1 = cv2.imread(src1 + fol1_img[i],0)
        img2 = cv2.imread(src2 + fol2_img[i],0)
        logging.debug("bit_AND : {0} AND {0}".format(fol1_img[i],fol2_img[i]))

        final_img = cv2.bitwise_and(img1,img2)
        cv2.imwrite(out + fol1_img[i], final_img)







class RangeFilter:

    # Given a folder directory,
    def sampleIntensity(self):
        print("Nothing here yet")
