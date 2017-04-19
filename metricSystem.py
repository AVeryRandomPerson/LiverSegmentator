import numpy as np
from CONSTANTS import *
import cv2
import os

def getAccuracy(annotate_img, result_img):
    accuracy = 0
    score_img = cv2.bitwise_xor(result_img, annotate_img)
    _, score_img = cv2.threshold(score_img, 80, 255, cv2.THRESH_BINARY)
    cv2.imshow('score',score_img)
    cv2.waitKey(0)
    total_cases = len(result_img) * len(result_img[0])
    errors = np.count_nonzero(score_img)
    accuracy = (total_cases - errors)*100 / total_cases
    return accuracy


def getTruePosRate(annotate_img, result_img):
    tp_rate = 0
    tp_img = cv2.bitwise_and(result_img, annotate_img)
    _, score_img = cv2.threshold(tp_img, 80, 255, cv2.THRESH_BINARY)
    cv2.waitKey(0)
    total_tp = np.count_nonzero(tp_img)
    total_p = np.count_nonzero(annotate_img)
    tp_rate = (total_tp/total_p) *100

    return tp_rate

def getTrueNegativeRate(annotate_img, result_img):
    tn_rate = 0
    tn_img = cv2.bitwise_or(result_img, annotate_img)
    _, score_img = cv2.threshold(tn_img, 80, 255, cv2.THRESH_BINARY)
    tn_img = cv2.bitwise_not(tn_img)

    cv2.waitKey(0)
    total_tn = np.count_nonzero(tn_img)
    total_n = np.count_nonzero(cv2.bitwise_not(annotate_img))
    tn_rate = (total_tn/total_n) *100

    return tn_rate

getSensitivity = getTruePosRate
getSpecificity = getTrueNegativeRate


GTs = 'C:/Users/acer/Desktop/TestSamples/LiverSegmentator/sourceCT/annotated_masks/'
Target = 'C:/Users/acer/Desktop/TestSamples/LiverSegmentator/all-datasets/5folds_16n8r_histEQ_gamma0f5/output/c1_103x103_lbp103x103/'

imgs = ['scan9','scan83','scan188','scan209','scan269']


def reportCSV():
    if(not os.path.exists(BASE_DIR + 'results.csv')):
        open(BASE_DIR + 'results.csv','a').close()

    try:
        with open(BASE_DIR + 'results.csv', 'w+') as report_file:
            report_file.writelines("full-dataset-name,folds,LBP-Rad,LBP-N,Gamma,useCannyEdge,useHistEQ,SlideWinSize,LSVC-C,F-centralDistance,F-stdDevCoeff,SENSITIVITY,SPECIFICITY,PRECISION,ACCURACY\n")
            dataset_set = os.listdir(BASE_DIR + 'all-datasets/')
            for dataset in dataset_set:
                model_execution_set = os.listdir(BASE_DIR + 'all-datasets/{0}/output'.format(dataset))
                for model_exc in model_execution_set:
                    fullname = dataset + model_exc
                    folds = dataset.split('folds_')[0]
                    lbp_rad = dataset.split('_')[1].split('n')[1].split('r')[0]
                    lbp_N = dataset.split('_')[1].split('n')[0]

                    gamma, useCanny, useHistEQ = _getPreProcessSettings(dataset)
                    dims, c, useSDV, useCCM = _getTrainingParams(model_exc)

                    sensitivity = 1.0
                    specificity = 1.0
                    precision = 1.0
                    accuracy = 1.0

                    report_file.writelines('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}\n'.format(
                        fullname,
                        folds,
                        lbp_rad,
                        lbp_N,
                        gamma,
                        useCanny,
                        useHistEQ,
                        dims,
                        c,
                        useSDV,
                        useCCM,
                        sensitivity,
                        specificity,
                        precision,
                        accuracy
                    ))





    except:
        raise Exception("Bad directory during database folder walking.")



def _getPreProcessSettings(dataset_name):
    params = dataset_name.split('_')
    gamma = '1.0'
    useCanny = 'False'
    useHistEQ = 'False'

    if 'gamma' in dataset_name:
        gamma = dataset_name.split('gamma')[1].split('_')[0].replace('f','.')
        print(gamma)

    if 'histEQ' in params:
        useHistEQ = 'True'

    if 'CannyEdge' in params:
        CannyEdge = 'True'


    return gamma, useCanny, useHistEQ


def _getTrainingParams(model_exc):
    params = model_exc.split('_')

    dims = params[1]
    c = params[0].split('c')[1].replace('f','.')
    useSDV = 'False'
    useCCM = 'False'

    if 'sdv' in model_exc:
        useSDV = 'True'

    if 'ccm' in model_exc:
        useCCM = 'True'

    return dims, c, useSDV, useCCM

if __name__ == '__main__':
    reportCSV()