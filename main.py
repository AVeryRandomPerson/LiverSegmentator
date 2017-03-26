from FuzzyOperator import FuzzyClusterer


PATH = "C:/Users/acer/Desktop/TestSamples/Mixed"
FILENAME = "/I0000049"
FORMAT = ".jpg"
IMG_LOC = PATH+FILENAME+FORMAT

OUT_FILENAME = FILENAME + "-RESULTSx"
OUT_FORMAT = ".png"
OUT_LOC = PATH+OUT_FILENAME+OUT_FORMAT

PARAMs = [(8, 0.000001, 5000, None)]


clusterer = FuzzyClusterer(PATH, FILENAME, FORMAT)
clusterer.cMeansIterative(PARAMs)
clusterer.saveAllResults()
