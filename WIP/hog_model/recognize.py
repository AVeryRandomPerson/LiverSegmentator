import cv2

from WIP.hog_model import hog

TRAINING_LIVER = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/liver/training/"
TRAINING_NONLIVER = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/non-liver/training/"
TESTING = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP/non-liver/testing/"

desc = hog.HistOrientGrad(9, (2, 2), (3, 3))
data = []
labels = []

farray =desc.describe(cv2.imread("C:/Users/acer/Desktop/TestSamples/ML-Dataset/CT_SCAN/testing/"+'scan6.jpg',0))
print(len(farray))
#img = farray.reshape((396,504))
#cv2.imshow('0',img)
cv2.waitKey(0)