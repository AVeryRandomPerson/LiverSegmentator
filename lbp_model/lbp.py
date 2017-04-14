import numpy as np
from skimage import feature


#   LocalBinaryPatterns
#       Class contains lbp parameters. Can describe an image.
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, mode="H", eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")

        if(mode == "H"):
            return self.computeHistogram(lbp, eps)

        else:
            return lbp

    def computeHistogram(self, lbpImage, eps=1e-7):
        (hist, _) = np.histogram(lbpImage.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalising the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist


