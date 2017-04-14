from lbp_model.core import Dataset
from lbp_model import lbp

descriptor = lbp.LocalBinaryPatterns(16,8)
dataset1 = Dataset('C:/Users/acer/Desktop/TestSamples/LiverSegmentator/', descriptor, 5)
#dataset1.trainDataset()
dataset1.lsvcPredictData(C=100)