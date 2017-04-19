import numpy as np
from CONSTANTS import *
import cv2
import os
from core import getTestingList

def getAccuracy(annotate_img, result_img):
    accuracy = 0
    score_img = cv2.bitwise_xor(result_img, annotate_img)
    _, score_img = cv2.threshold(score_img, 80, 255, cv2.THRESH_BINARY)

    total_cases = len(result_img) * len(result_img[0])
    errors = np.count_nonzero(score_img)
    accuracy = (total_cases - errors)*100 / total_cases
    return accuracy


def getTruePosRate(annotate_img, result_img):
    tp_rate = 0
    tp_img = cv2.bitwise_and(result_img, annotate_img)
    _, score_img = cv2.threshold(tp_img, 80, 255, cv2.THRESH_BINARY)

    total_tp = np.count_nonzero(tp_img)
    total_p = np.count_nonzero(annotate_img)
    tp_rate = (total_tp/total_p) *100

    return tp_rate

def getTrueNegativeRate(annotate_img, result_img):
    tn_rate = 0
    tn_img = cv2.bitwise_or(result_img, annotate_img)
    _, score_img = cv2.threshold(tn_img, 80, 255, cv2.THRESH_BINARY)
    tn_img = cv2.bitwise_not(tn_img)

    total_tn = np.count_nonzero(tn_img)
    total_n = np.count_nonzero(cv2.bitwise_not(annotate_img))
    tn_rate = (total_tn/total_n) *100

    return tn_rate


def getConfMatrix(gt_img_path, generated_img_path):
    gt_img = cv2.imread(gt_img_path,0)
    gen_img = cv2.imread(generated_img_path,0)

    total_cases = len(gt_img) * len(gt_img[0])
    total_T = np.count_nonzero(gt_img)
    total_N = total_cases - np.count_nonzero(gt_img)

    tn_img = cv2.bitwise_or(gen_img, gt_img)
    _, tn_img = cv2.threshold(tn_img, 80, 255, cv2.THRESH_BINARY)
    tn_img = cv2.bitwise_not(tn_img)
    total_TN = np.count_nonzero(tn_img)
    del tn_img


    _, tp_img = cv2.threshold(cv2.bitwise_and(gen_img, gt_img), 80, 255, cv2.THRESH_BINARY)
    total_TP = np.count_nonzero(tp_img)
    del tp_img

    _, t_img = cv2.threshold(cv2.bitwise_xor(gen_img, gt_img), 80, 255, cv2.THRESH_BINARY)
    errors = np.count_nonzero(t_img)
    del t_img

    total_FP = total_N - total_TN
    total_FN = total_T - total_TP

    if(total_TP + total_FP + total_TN + total_FN != total_cases):
        raise Exception("Wrong formula for confusion Matrix")
    return total_TP, total_FP, total_TN, total_FN, total_cases, total_T, total_N


def getScoreFolder(gt_fol_dir, gen_img_dir, folds):
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0
    total_cases = 0
    total_T = 0
    total_N = 0

    img_name_list = getTestingList(BASE_DIR + 'sourceCT/kfolds_list/{0}folds_test_list.txt'.format(folds))

    for img_name in img_name_list:
        tp, fp, tn, fn, tc, t, n = getConfMatrix(gt_fol_dir + img_name + '.png', gen_img_dir + img_name + '.jpg')
        total_TP = total_TP + tp
        total_FP = total_FP + fp
        total_TN = total_TN + tn
        total_FN = total_FN + fn
        total_cases = total_cases + tc
        total_T = total_T + t
        total_N = total_N + n


    sensitivity = total_TP/total_T
    specificity = total_TN/total_N
    accuracy = (total_TN + total_TP)/total_cases

    if (total_TP + total_FP == 0):
        precision = 'NaN'
    else:
        precision = total_TP / (total_TP + total_FP)

    return sensitivity, specificity, precision, accuracy

getSensitivity = getTruePosRate
getSpecificity = getTrueNegativeRate


GTs = 'C:/Users/acer/Desktop/TestSamples/LiverSegmentator/sourceCT/annotated_masks/'
Target = 'C:/Users/acer/Desktop/TestSamples/LiverSegmentator/all-datasets/5folds_16n8r_histEQ_gamma0f5/output/c1_103x103_lbp103x103/'

imgs = ['scan9','scan83','scan188','scan209','scan269']


def reportCSV():
    if(not os.path.exists(BASE_DIR + 'results.csv')):
        open(BASE_DIR + 'results.csv','a').close()

    with open(BASE_DIR + 'results.csv', 'w+') as report_file:
        report_file.writelines("full-dataset-name,folds,LBP-Rad,LBP-N,Gamma,useCannyEdge,useHistEQ,SlideWinSize,LSVC-C,F-stdDevCoeff,F-centralDistance,SENSITIVITY,SPECIFICITY,PRECISION,ACCURACY\n")
        dataset_set = os.listdir(BASE_DIR + 'all-datasets/')
        for dataset in dataset_set:
            model_execution_set = os.listdir(BASE_DIR + 'all-datasets/{0}/output/'.format(dataset))
            for model_exc in model_execution_set:
                fullname = dataset + model_exc
                folds = dataset.split('folds_')[0]
                lbp_rad = dataset.split('_')[1].split('n')[1].split('r')[0]
                lbp_N = dataset.split('_')[1].split('n')[0]

                gamma, useCanny, useHistEQ = _getPreProcessSettings(dataset)
                dims, c, useSDV, useCCM = _getTrainingParams(model_exc)

                sensitivity, specificity, precision, accuracy = getScoreFolder(BASE_DIR + 'sourceCT/annotated_masks/',
                                                                               BASE_DIR + 'all-datasets/{0}/output/{1}/'.format(dataset,model_exc),
                                                                               folds)

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


def _getPreProcessSettings(dataset_name):
    params = dataset_name.split('_')
    gamma = '1.0'
    useCanny = 'False'
    useHistEQ = 'False'


    if 'gamma' in dataset_name:
        gamma = dataset_name.split('gamma')[1].split('_')[0].replace('f','.')


    if 'histEQ' in dataset_name:
        useHistEQ = 'True'

    if 'CannyEdge' in dataset_name:
        useCanny = 'True'


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