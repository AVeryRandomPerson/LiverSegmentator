import cv2
import numpy as np


# I plan to use this as the image class that manages the image data.
# Hence processing related to images should be performed with methods in this class alone.
class CTImage:
    def __init__(self,location):
        self.image = cv2.imread(location , 0)
        self.width = len(self.image[0])
        self.height = len(self.image)
        self.image_dimension = (self.width,self.height)

    def getImage(self):
        return self.image

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getImageDimension(self):
        return self.image_dimension

    #Processes each pixel of image into a flat array of intensity data.
    def getIntensityData(self):
        '''
        xpts = np.zeros(0)
        for y in range(0, self.height):
            for x in range(0, self.width):
                xpts = np.hstack((xpts, self.image[y][x]))
                print('{0} , {1}'.format(x, y))
        ypts = xpts'''
        xpts = np.ravel(self.image)
        ypts = xpts
        return np.vstack((xpts, ypts))
