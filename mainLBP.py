from lbp_model.core import Dataset
from lbp_model import lbp
from CONSTANTS import *

#   basicTrainingProgramme(1D-ARRAY<Float> cList,
#                          STRING dsBaseDir,
#                          INT descNPoints,
#                          INT descRadius,
#                          INT folds,
#                          BOOL useSobel,
#                          FLOAT gamma,
#                          BOOL useHistEQ,
#                          BOOL useSDV,
#                          BOOL useCCostMeasure,
#                          TUPLE(int,int) tile_dimensions)
#   With the given parameters, generates a dataset, preprocesses it, trains it(if necessary) and predicts test images with it.
#   Binaries and Predicted Images may be generated and written during the process. But the function itself returns NONE.
#
#   <Input>
#       required 1D-ARRAY<Float> cList | the list of C parameters of the SVC. | Future : May implement more control parameters.
#       required STRING dsBaseDir | the dataset base directory.
#       required INT descNPoints | the number of points of the lbp descriptor. [See Class Local Binary Patterns]
#       required INT folds | the k value in the kfolds cross validation used to partition the data.
#       required BOOL useSobel | the decision to use sobel filters during preprocessing.
#       required FLOAT gamma | the gamma value of the contrast adjustment preprocessing. G= 1.0 makes no difference.
#       required BOOL useHistEQ | the decision to use histogram equilization preprocessing.
#       required BOOL use SDV | the decision to use standard deviation as feature for SVC.
#       required BOOL use CCostMeasure | the decision to use distance from center cost metrics as feature for SVC.
#       optional TUPLE(int, int) tile_dimensions | the size of the sliding window tile for training and testing. Format (X by Y) | Default (73, 73).
#   <Output>
#       NONE.
def basicTrainingProgramme(cList, dsBaseDir, descNPoints, descRadius, folds, useSobel, gamma, useHistEQ, useSDV, useCCostMeasure, tile_dimensions=(73,73)):
    dataset = Dataset(dsBaseDir, lbp.LocalBinaryPatterns(descNPoints, descRadius), folds, useSobel, gamma, useHistEQ)
    if(not dataset.hasTrainedBinaries()):
       dataset.trainDataset(tile_dimensions=tile_dimensions, useSDV=useSDV, useCCostMeasure=useCCostMeasure)

    for c in cList:
        dataset.lsvcPredictData(C=c, tile_dimensions=tile_dimensions, useSDV=useSDV,useCCostMeasure=useCCostMeasure)

# Entry point of entire lbp model program.
if __name__ == '__main__':

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
