import numpy as np
from skimage import feature


#   LocalBinaryPatterns
#       Class contains lbp parameters. Can describe an image.
#       INT numPoints   [Number of neighbourhood points to capture]
#       INT radius      [Value of lbp region radius]
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    #   describe(NUMPY_ARRAY[Image] image, STRING mode, Float eps)
    #       computes lbp features of a given image. Parameters are based on the object's attributes.
    #
    #
    #   <Input>
    #       required NUMPY_ARRAY[Image] image | the Numpy array representation of the image.
    #       required STRING mode | determines the output returned. See below for available modes.
    #       optional FLOAT eps | the eps value used when computing histogram.
    #   <Output>
    #       mode == "H"
    #          1D-ARRAY<int> /anonymous/ | the histogram representation of the lbp image.
    #
    #       mode == *
    #           NUMPY_ARRAY[Image] lbp | the lbp image.
    def describe(self, image, mode="H", eps=1e-7, chingchong=None):
        lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")

        if(mode == "H"):
            return self.computeHistogram(lbp, eps)

        else:
            return lbp

    #   computeHistogram(NUMPY_ARRAY[Image] lbpImage, Float eps)
    #       computes the lbp histogram based on the given lbp image.
    #
    #
    #   <Input>
    #       required NUMPY_ARRAY[Image] image | the Numpy array representation of the image.
    #       optional FLOAT eps | the eps value used when computing histogram.
    #   <Output>
    #       mode == "H"
    #          1D-ARRAY<int> hist | the histogram representation of the lbp image.
    def computeHistogram(self, lbpImage, eps=1e-7):
        (hist, _) = np.histogram(lbpImage.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalising the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist
