import numpy as np
from lbp_model import core
from skimage import feature



class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, mode="H", eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, method="uniform")

        if(mode == "H"):
            return self.computeHistogram(lbp)

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


if __name__ == "__main__":
    import cv2
    from imutils import paths



    descriptor = LocalBinaryPatterns(24, 8)
    exp_path = "C:/Users/acer/Desktop/TestSamples/LiverSegmentator/all-datasets/5folds_24n8r/preprocessed_samples/"
    for img_path in paths.list_images("C:/Users/acer/Desktop/TestSamples/LiverSegmentator/sourceCT/CTs/"):
        img = cv2.imread(img_path,0)
        img_name = img_path.split('/').pop()
        lbp_img = descriptor.describe(img, mode='I')
        cv2.imwrite(exp_path + img_name, lbp_img)
        print("Exported" + exp_path + img_name)
        

