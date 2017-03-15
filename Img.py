import cv2
import numpy as np


# I plan to use this as the image class that manages the image data.
# Hence processing related to images should be performed with methods in this class alone.
class CTImage:
    def __init__(self,path,name,format):
        self.path = path
        self.name = name
        self.format = format

        img_loc = path+name+format
        self.image = cv2.imread(img_loc , 0)
        self.width = len(self.image[0])
        self.height = len(self.image)

    #Processes each pixel of image into a flat array of intensity data.
    def getIntensityData(self):
        xpts = np.ravel(self.image)
        ypts = xpts
        return np.vstack((xpts,ypts))


# Class storing respective results
'''
iCENTER = 0
iLABELS = 1
iINIT_PARTITION = 2
iFIN_EUCLID = 3
iOBJ_HIST = 4
iITERS_EXEC = 5
iPART_COEFF = 6
iCOL_SCHEME = 7
'''
class ClusterResult:
    def __init__(self,results,path,name,format):
        self.center = results[0]
        self.labels = results[1]
        self.init_partition = results[2]
        self.fin_euclid = results[3]
        self.obj_hist = results[4]
        self.iters_exec = results[5]
        self.part_coeef = results[6]
        self.col_scheme = results[7]
        self.src_path = path
        self.src_name = name
        self.src_format = format



