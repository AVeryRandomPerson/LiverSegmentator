import cv2
import numpy as np

# OBSOLETE MODULE. Will be removed in the future.

# image class storing relevant image information
class CTImage:
    def __init__(self, path, name, extention):
        self.path = path
        self.name = name
        self.extention = extention

        img_loc = path+name + extention
        self.image = cv2.imread(img_loc, 0)
        self.width = len(self.image[0])
        self.height = len(self.image)

    # Processes each pixel of image into a flat array of intensity data.
    def getIntensityData(self):
        xpts = np.ravel(self.image)
        return np.vstack((xpts, xpts))


