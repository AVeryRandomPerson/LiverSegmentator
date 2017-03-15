import skfuzzy as fuzz
import numpy as np
import colorsys
import math
import cv2
import Img
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
    def __init__(self,path,name,format):
        self.ct_image = Img.CTImage(path,name,format)
        self.img_data = self.ct_image.getIntensityData()
        self.results = []


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

        # Saving Results
        result = Img.ClusterResult((center,
                                    labels,
                                    init_partition,
                                    fin_euclid,
                                    obj_hist,
                                    iters_exec,
                                    part_coeff,
                                    color_scheme),
                                   self.ct_image.getPath(),
                                   self.ct_image.getName(),
                                   self.ct_image.getFormat())
        self.results.append(result)


    #Iteratively runs CMeansClustering with various parameters
    def cMeansIterative(self,paramsList):
        for i in range(0,len(paramsList)):
            print(paramsList[i][0])
            self.cMeans(paramsList[i])


    # Resets the results which have been computed.
    def resetResults(self):
        self.results = []


    # Removes the current image which is being referenced.
    def resetImage(self):
        self.img_data = np.zeros(0)


    # Changes the image source which is being referenced.
    def changeImage(self,img_loc):
        ct_image = Img.CTImage(img_loc)
        self.img_data = ct_image.getIntensityData()


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
    def computeClusteredImage(self):
        self.computeClusteredImage(-1)


    # Processes and acquires the clustered image from results.
    # Latest result will be used if no reference is made for previous results
    def computeClusteredImage(self, targetIndex):
        #Use the latest result appended if no index given
        if(targetIndex < 0):
            index = (len(self.results)-1)

        else:
            index = targetIndex

        clustered_image = np.zeros((self.ct_image.getWidth(), self.ct_image.getHeight(), 3))
        x = 0
        y = 0

        labels = self.results[index][iLABELS]
        rgb_scheme = self.results[index][iCOL_SCHEME]
        for i in range(0, len(labels)):
            clustered_image[y][x][R] = rgb_scheme[labels[i]][R]
            clustered_image[y][x][G] = rgb_scheme[labels[i]][G]
            clustered_image[y][x][B] = rgb_scheme[labels[i]][B]
            x = x + 1
            if (x == self.ct_image.getWidth()):
                x = 0
                y = y + 1

        return clustered_image

    # Saves the current result out as an image
    def saveResult(self):
        out_loc = self.ct_image.getPath()+self.ct_image.getName()+'.png'
        cv2.imwrite(out_loc,self.computeClusteredImage())

    # Saves all the results out as images
    def saveAllResults(self):
        for i in range(0,len(self.results)):
            out_loc = self.ct_image.getPath()+self.ct_image.getName()+'['+str(i)+']'+'.png'
            cv2.imwrite(out_loc,self.computeClusteredImage(i))


    # Returns all the results
    def getResults(self):
        return self.results


    # Returns the latest results
    def getLatestResult(self):
        return self.results[len(self.results) - 1]




