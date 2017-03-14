import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import colorsys
import math
import cv2
from ImgProcessor import CTImage

SAT = 1.0
VAL = 1.0
PATH = "C:/Users/acer/Desktop/TestSamples"
FILENAME = "/I0001014.jpg"

NCENTERS = 4
ERROR = 0.005
MAXITER = 1000
INIT = None

ct_image = CTImage(PATH+FILENAME)
clustered_image = np.zeros((ct_image.getWidth(), ct_image.getHeight(), 3))
alldata = ct_image.getIntensityData()

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    alldata, NCENTERS, 2, error=ERROR, maxiter=MAXITER, init=INIT)

labels = np.argmax(u,axis=0)

print("Loading Processed Image...")

rgb_scheme = [[0, 0, 0], [255, 255, 255]]
if (NCENTERS > 2):
    colour_distributions = NCENTERS - 2
    for i in range(0, colour_distributions):
        hue_percentile = (1.0 / colour_distributions) * (i + 1.0)
        rgb = list(colorsys.hsv_to_rgb(hue_percentile, SAT, VAL))
        rgb[0] = math.ceil(rgb[0] * 255)
        rgb[1] = math.ceil(rgb[1] * 255)
        rgb[2] = math.ceil(rgb[2] * 255)

        rgb_scheme.append(rgb)

x=0
y=0
for i in range(0,len(labels)):
    clustered_image[y][x][0] = rgb_scheme[labels[i]][0]
    clustered_image[y][x][1] = rgb_scheme[labels[i]][1]
    clustered_image[y][x][2] = rgb_scheme[labels[i]][2]
    x = x + 1
    if (x == ct_image.getWidth()):
        x = 0
        y = y + 1


cv2.imwrite("C:/Users/acer/Desktop/TestSamples/I0001014-RESULTSx.png", clustered_image)
cv2.imshow("Clustered Image",mat=clustered_image)