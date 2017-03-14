import skfuzzy as fuzz
import numpy as np
import colorsys
import math
from ImgProcessor import CTImage
#Colour const
SAT = 1.0
VAL = 1.0

#Colour index
R = 2
G = 1
B = 0

#Parameter Indexes
iNCENTERS = 0
iERROR = 1
iMAXITER = 2
iINIT = 3

#Result Indexes
iCENTER = 0
iLABELS = 1
iINIT_PARTITION = 2
iFIN_EUCLID = 3
iOBJ_HIST = 4
iITERS_EXEC = 5
iPART_COEFF = 6
iCOL_SCHEME = 7


# Class handling the fuzzy c-means clustering and visualizing its results
class FuzzyClusterer():
    width = 0
    height = 0
    results = []
    img_data = np.zeros(0)


    def __init__(self,img_loc):
        ct_image = CTImage(img_loc)
        self.img_data = ct_image.getIntensityData()
        self.width = ct_image.getWidth()
        self.height = ct_image.getHeight()

    #Iteratively runs CMeansClustering with various parameters
    def cMeansIterative(self,paramsList):
        for i in range(0,len(paramsList)):
            self.cMeans(paramsList[0])

        return self.results

    #Basic cMeans operations using skfuzzy.
    def cMeans(self,params):
        center, fin_partition, init_partition, fin_euclid, obj_hist, iters_exec, part_coeff = fuzz.cluster.cmeans(
            data = self.img_data,
            c=params[iNCENTERS],
            m=3,
            error=params[iERROR],
            maxiter=params[iMAXITER],
            init=params[iINIT])

        # We use labels instead of the U matrix given by the c-partition.
        labels = np.argmax(fin_partition, axis=0)

        # Compute the colour scheme for this set of cluster
        color_scheme = self._genColourScheme(params[iNCENTERS])
        self.results.append((center, labels, init_partition, fin_euclid, obj_hist, iters_exec, part_coeff, color_scheme))
        return center, labels, init_partition, fin_euclid, obj_hist, iters_exec, part_coeff, color_scheme

    # Resets the results which have been computed.
    def resetResults(self):
        self.results = []

    # Removes the current image which is being referenced.
    def resetImage(self):
        self.img_data = np.zeros(0)
        self.width = 0
        self.height = 0

    # Changes the image source which is being referenced.
    def changeImage(self,img_loc):
        ct_image = CTImage(img_loc)
        self.img_data = ct_image.getIntensityData()
        self.width = ct_image.getWidth()
        self.height = ct_image.getHeight()

    # Resets the whole class as if it is a new object.
    # If no image is specified when this is called, the old image will stay referenced.
    def resetClusterer(self,img_loc = None):
        self.resetResults()

        if(not img_loc):
            self.resetImage()

        else:
            self.changeImage(self,img_loc)

    # Processes and acquires the clustered image from results.
    # Latest result will be used if no reference is made for previous results
    def computeClusteredImage(self,resultIndex=None):
        #Use the latest result appended if no index given
        if(not resultIndex):
            resultIndex = (len(self.results)-1)

        clustered_image = np.zeros((self.width, self.height, 3))
        x = 0
        y = 0

        labels = self.results[resultIndex][iLABELS]
        rgb_scheme = self.results[resultIndex][iCOL_SCHEME]
        for i in range(0, len(labels)):
            clustered_image[y][x][R] = rgb_scheme[labels[i]][R]
            clustered_image[y][x][G] = rgb_scheme[labels[i]][G]
            clustered_image[y][x][B] = rgb_scheme[labels[i]][B]
            x = x + 1
            if (x == self.width):
                x = 0
                y = y + 1

        return clustered_image

    #Generates a colour scheme of RGB based on the percentile of RGB values and number of clusters.
    def _genColourScheme(self, ncenters):
        #Black and white if only 2 cluster.
        rgb_scheme = [[0, 0, 0], [255, 255, 255]]
        if (ncenters > 2):
            colour_distributions = ncenters - 2
            for i in range(0, colour_distributions):
                hue_percentile = (1.0 / colour_distributions) * (i + 1.0)
                rgb = list(colorsys.hsv_to_rgb(hue_percentile, SAT, VAL))
                rgb[R] = math.ceil(rgb[R] * 255)
                rgb[G] = math.ceil(rgb[G] * 255)
                rgb[B] = math.ceil(rgb[B] * 255)
                rgb_scheme.append(rgb)

        return rgb_scheme

