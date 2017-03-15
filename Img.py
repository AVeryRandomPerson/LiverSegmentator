import cv2
import numpy as np


# I plan to use this as the image class that manages the image data.
# Hence processing related to images should be performed with methods in this class alone.
class CTImage:
    def __init__(self,path,name,format):
        self.img_path = path
        self.img_name = name
        self.img_format = format

        img_loc = path+name+format
        self.image = cv2.imread(img_loc , 0)
        self.width = len(self.image[0])
        self.height = len(self.image)

    #Returns the path of the image
    def getPath(self):
        return self.img_path

    #Returns the image name
    def getName(self):
        return self.img_name

    #Returns the image format
    def getFormat(self):
        return self.img_format

    #Returns the numpy array representing the image
    def getImage(self):
        return self.image

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    #Returns the tuple of (Width,Height) of the image
    def getImageDimension(self):
        return (self.width,self.height)

    #Processes each pixel of image into a flat array of intensity data.
    def getIntensityData(self):
        xpts = np.ravel(self.image)
        ypts = xpts
        return np.vstack((xpts,ypts))


class ClusterResult:
    # Result Indexes
    iCENTER = 0
    iLABELS = 1
    iINIT_PARTITION = 2
    iFIN_EUCLID = 3
    iOBJ_HIST = 4
    iITERS_EXEC = 5
    iPART_COEFF = 6
    iCOL_SCHEME = 7

    def __init__(self,results,path,name,format):
        self.center = results[ClusterResult.iCENTER]
        self.labels = results[ClusterResult.iLABELS]
        self.init_partition = results[ClusterResult.iINIT_PARTITION]
        self.fin_euclid = results[ClusterResult.iFIN_EUCLID]
        self.obj_hist = results[ClusterResult.iOBJ_HIST]
        self.iters_exec = results[ClusterResult.iITERS_EXEC]
        self.part_coeef = results[ClusterResult.iPART_COEFF]
        self.col_scheme = results[ClusterResult.iCOL_SCHEME]
        self.src_path = path
        self.src_name = name
        self.src_format = format



