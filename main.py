import cv2
from FuzzyOperator import FuzzyClusterer


PATH = "C:/Users/acer/Desktop/TestSamples"
FILENAME = "/I0001014"
FORMAT = ".jpg"
IMG_LOC = PATH+FILENAME+FORMAT

OUT_FILENAME = FILENAME + "-RESULTSx"
OUT_FORMAT = ".png"
OUT_LOC = PATH+OUT_FILENAME+OUT_FORMAT

PARAMs = [(2,0.005,1000,None),(3,0.005,1000,None),(4,0.005,1000,None)]


clusterer = FuzzyClusterer(PATH,FILENAME,FORMAT)
clusterer.cMeansIterative(PARAMs)
clusterer.saveAllResults()


