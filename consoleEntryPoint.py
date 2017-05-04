import argparse
from mainLBP import runTrainingProgramme
from CONSTANTS import BASE_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cList", help="A list of C parameters for the SVM training.",
                        type=int, nargs='+')

    parser.add_argument("--lbpNP", help="The number of Neighborhood points of the LBP feature.",
                        type=int)

    parser.add_argument("--lbpRad", help="The radius of the LBP feature.",
                        type=int)

    parser.add_argument("--folds", help="The k value of the k-folds partitioning of data.",
                        type=int)

    parser.add_argument("--canny", help="The decision to use CannyEdge preprocessing.",
                        action='store_true')

    parser.add_argument("--gamma", help="The value of gamma in gamma correction preprocessing. 1.0 = NO CHANGE.",
                        type=float)

    parser.add_argument("--histEQ", help="The decision to use histogram Equalization preprocessing.",
                        action='store_true')

    parser.add_argument("--sdv", help="The decision to use standard deviation as feature.",
                        action='store_true')

    parser.add_argument("--euclidDist", help="The decision to use Euclidian Distance of pixels from approximate centre as feature.",
                        action='store_true')

    parser.add_argument("--tileW", help="The width of sliding window.",
                        type=int)

    parser.add_argument("--tileH", help="The height of sliding window.",
                        type=int)

    args = parser.parse_args()
    print("executing", args)
    runTrainingProgramme(cList=args.cList,
                         dsBaseDir=BASE_DIR,
                         descNPoints=args.lbpNP,
                         descRadius=args.lbpRad,
                         folds=args.folds,
                         useCannyEdge=args.canny,
                         gamma=args.gamma,
                         useHistEQ=args.histEQ,
                         useSDV=args.sdv,
                         useCCostMeasure=args.euclidDist,
                         tile_dimensions=(args.tileW, args.tileH))


    #Example Usage
    #
    #
    # CMD console
    # C:/>
    # C:/>Python C:/Users/acer/PycharmProjects/LiverSegmentator/consoleEntryPoint.py --cList 100 1 --lbpNP 16 --lbpRad 8 --folds 5 --canny --gamma 1.5 --histEQ --sdv --tileW 103 --tileH 103

'''
##From mainLBP.py
if __name__ == '__main__':

    runTrainingProgramme(cList = [0.001, 0.01, 1 , 100, 1000],
                         dsBaseDir=BASE_DIR,
                         descNPoints=16,
                         descRadius=8,
                         folds=5,
                         useCannyEdge=False,
                         gamma=1.0,
                         useHistEQ=False,
                         useSDV=True,
                         useCCostMeasure=True ,
                         tile_dimensions=(73, 73))


    runTrainingProgramme(cList = [0.001, 0.01, 1 , 100, 1000],
                         dsBaseDir=BASE_DIR,
                         descNPoints=16,
                         descRadius=8,
                         folds=5,
                         useCannyEdge=True,
                         gamma=1.0,
                         useHistEQ=True,
                         useSDV=True,
                         useCCostMeasure=True ,
                         tile_dimensions=(73, 73))

'''