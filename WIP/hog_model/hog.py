import numpy as np
from skimage import feature

class HistOrientGrad:
    def __init__(self, orients, ppc, cpb):
        self.orients = orients
        self.ppc = ppc
        self.cpb = cpb


    def describe(self, image):
        farray = feature.hog(image, self.orients,self.ppc, self.cpb, visualise= False, feature_vector=True)

        return farray