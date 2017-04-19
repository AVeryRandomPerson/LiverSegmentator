import cv2

img = cv2.imread('C:/Users/acer/Desktop/TestSamples/LiverSegmentator/sourceCT/annotated_masks/scan8.png',0)
cv2.imshow('name',img)
cv2.waitKey(0)