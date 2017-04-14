from lbp_model.core import Dataset
from lbp_model import lbp

BASE_DIR = 'C:/Users/acer/Desktop/TestSamples/LiverSegmentator/'

descriptor = lbp.LocalBinaryPatterns(8, 8)
dataset2 = Dataset(BASE_DIR, descriptor, 5)
dataset2.trainDataset()
dataset2.lsvcPredictData(C=1000)
dataset2.lsvcPredictData(C=100)
dataset2.lsvcPredictData(C=1)
dataset2.lsvcPredictData(C=0.01)
dataset2.lsvcPredictData(C=0.001)

del dataset2
del descriptor

descriptor = lbp.LocalBinaryPatterns(16, 8)
dataset1 = Dataset(BASE_DIR, descriptor, 5)
# dataset1.trainDataset()
dataset1.lsvcPredictData(C=1000)
dataset1.lsvcPredictData(C=1)
dataset1.lsvcPredictData(C=0.01)
dataset1.lsvcPredictData(C=0.001)

del dataset1
del descriptor

descriptor = lbp.LocalBinaryPatterns(12, 8)
dataset2 = Dataset(BASE_DIR, descriptor, 5)
dataset2.trainDataset()
dataset2.lsvcPredictData(C=1000)
dataset2.lsvcPredictData(C=100)
dataset2.lsvcPredictData(C=1)
dataset2.lsvcPredictData(C=0.01)
dataset2.lsvcPredictData(C=0.001)

del dataset2
del descriptor


descriptor = lbp.LocalBinaryPatterns(20, 8)
dataset2 = Dataset(BASE_DIR, descriptor, 5)
dataset2.trainDataset()
dataset2.lsvcPredictData(C=1000)
dataset2.lsvcPredictData(C=100)
dataset2.lsvcPredictData(C=1)
dataset2.lsvcPredictData(C=0.01)
dataset2.lsvcPredictData(C=0.001)

del dataset2
del descriptor


descriptor = lbp.LocalBinaryPatterns(24, 8)
dataset2 = Dataset(BASE_DIR, descriptor, 5)
dataset2.trainDataset()
dataset2.lsvcPredictData(C=1000)
dataset2.lsvcPredictData(C=100)
dataset2.lsvcPredictData(C=1)
dataset2.lsvcPredictData(C=0.01)
dataset2.lsvcPredictData(C=0.001)

del dataset2
del descriptor
