import numpy as np
from skimage import feature

class HistOrientGrad:
    def __init__(self, orients, ppc, cpb):
        self.orients = orients
        self.ppc = ppc
        self.cpb = cpb


    def describe(self, image):
        hist, img = feature.hog(image, self.orients,self.ppc, self.cpb, visualise= True, feature_vector=True)

        return hist, img