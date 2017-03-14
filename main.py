import cv2
from FuzzyOperator import FuzzyClusterer


PATH = "C:/Users/acer/Desktop/TestSamples"
FILENAME = "/I0001014"
FORMAT = ".jpg"
IMG_LOC = PATH+FILENAME+FORMAT

OUT_FILENAME = FILENAME + "-RESULTSx"
OUT_FORMAT = ".png"
OUT_LOC = PATH+OUT_FILENAME+OUT_FORMAT

NCENTERS = 4
ERROR = 0.005
MAXITER = 1000
INIT = None

clusterer = FuzzyClusterer(IMG_LOC)
clusterer.cMeans((NCENTERS,ERROR,MAXITER,INIT))
clustered_image = clusterer.computeClusteredImage()
print("Loading Processed Image...")

cv2.imwrite(OUT_LOC, clustered_image)
cv2.imshow("Clustered Image",mat=clustered_image)
cv2.waitKey(-1)

