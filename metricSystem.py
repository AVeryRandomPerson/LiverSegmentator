import numpy as np
from CONSTANTS import *
import cv2
import os
from core import getTestingList

#   getAccuracy(NUMPY_ARRAY[Image] ground_truth_img, NUMPY_ARRAY[Image] generated_img)
#   Performs bitwise computations between the images to compute accuracy scores.
#
#
#   <Input>
#       required NUMPY_ARRAY[Image] ground_truth_img | A Binary 2D-NUMPY_ARRAY representation of a ground truth annotation.
#       required NUMPY_ARRAY[Image] generated_img | A Binary 2D-NUMPY_ARRAY representation of an image to be compared with the ground truth
#   <Output>
#       FLOAT accuracy | The percentage score of Accuracy of the ground_truth_img from the generated_img.
def getAccuracy(ground_truth_img, generated_img):
    accuracy = 0
    score_img = cv2.bitwise_xor(generated_img, ground_truth_img)
    _, score_img = cv2.threshold(score_img, 80, 255, cv2.THRESH_BINARY)

    total_cases = len(generated_img) * len(generated_img[0])
    errors = np.count_nonzero(score_img)
    accuracy = (total_cases - errors)*100 / total_cases
    return accuracy


#   getTruePosRate(NUMPY_ARRAY[Image] ground_truth_img, NUMPY_ARRAY[Image] generated_img)
#   Performs bitwise computations between the images to compute sensitivity scores.
#
#
#   <Input>
#       required NUMPY_ARRAY[Image] ground_truth_img | A Binary 2D-NUMPY_ARRAY representation of a ground truth annotation.
#       required NUMPY_ARRAY[Image] generated_img | A Binary 2D-NUMPY_ARRAY representation of an image to be compared with the ground truth
#   <Output>
#       FLOAT accuracy | The percentage score of Sensitivity of the ground_truth_img from the generated_img.
def getTruePosRate(ground_truth_img, generated_img):
    tp_rate = 0
    tp_img = cv2.bitwise_and(generated_img, ground_truth_img)
    _, score_img = cv2.threshold(tp_img, 80, 255, cv2.THRESH_BINARY)

    total_tp = np.count_nonzero(tp_img)
    total_p = np.count_nonzero(ground_truth_img)
    tp_rate = (total_tp/total_p) *100

    return tp_rate

#   getTrueNegativeRate(NUMPY_ARRAY[Image] ground_truth_img, NUMPY_ARRAY[Image] generated_img)
#   Performs bitwise computations between the images to compute specificity scores.
#
#
#   <Input>
#       required NUMPY_ARRAY[Image] ground_truth_img | A Binary 2D-NUMPY_ARRAY representation of a ground truth annotation.
#       required NUMPY_ARRAY[Image] generated_img | A Binary 2D-NUMPY_ARRAY representation of an image to be compared with the ground truth
#   <Output>
#       FLOAT accuracy | The percentage score of Specificity of the ground_truth_img from the generated_img.
def getTrueNegativeRate(ground_truth_img, generated_img):
    tn_rate = 0
    tn_img = cv2.bitwise_or(generated_img, ground_truth_img)
    _, score_img = cv2.threshold(tn_img, 80, 255, cv2.THRESH_BINARY)
    tn_img = cv2.bitwise_not(tn_img)

    total_tn = np.count_nonzero(tn_img)
    total_n = np.count_nonzero(cv2.bitwise_not(ground_truth_img))
    tn_rate = (total_tn/total_n) *100

    return tn_rate


#   getConfMatrix(STRING gt_img_path, STRING generated_img_path)
#   Performs bitwise computations when necessary to acquire the confusion matrix. The values are returned with multiple return values.
#
#
#   <Input>
#       required STRING gt_img_path | The location path of a ground truth binary image.
#       required STRING generated_img_path | The location path of a generated binary image.
#   <Output>
#       INT total_TP | The total occurances of true positive classification between the ground truth and generated binary images.
#       INT total_FP | The total occurances of false positive classification between the ground truth and generated binary images.
#       INT total_TN | The total occurances of true negative classification between the ground truth and generated binary images.
#       INT total_FN | The total occurances of false negative classification between the ground truth and generated binary images.
#       INT total_cases | The total instances of classification.
#       INT total_T | The total occurances of True values in the Ground Truth.
#       INT total_F| The total occurances of False values in the Ground Truth.
def getConfMatrix(gt_img_path, generated_img_path):
    gt_img = cv2.imread(gt_img_path,0)
    gen_img = cv2.imread(generated_img_path,0)

    total_cases = len(gt_img) * len(gt_img[0])
    total_T = np.count_nonzero(gt_img)
    total_F = total_cases - np.count_nonzero(gt_img)

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

    total_FP = total_F - total_TN
    total_FN = total_T - total_TP

    if(total_TP + total_FP + total_TN + total_FN != total_cases):
        raise Exception("Wrong formula for confusion Matrix")
    return total_TP, total_FP, total_TN, total_FN, total_cases, total_T, total_F

#   getDatasetScore(STRING gt_fol_dir, STRING gen_img_dir, INT folds)
#   Iteratively compares each pair of images between the annotation binary source folder and the generated results source folder.
#   Resulting scores are over the total instances of all 5 images. Multiple return values.
#
#
#   <Input>
#       required STRING gt_img_path | The location path of a ground truth binary image.
#       required STRING generated_img_path | The location path of a generated binary image.
#       required INT folds | The dataset number of folds. (The k value used during partition of the data). Used to locate the list of test filenames.
#   <Output>
#       FLOAT sensitivity | The total sensitivity of the classification across all the generated images.
#       FLOAT specificity | The total specificity of the classification across all the generated images.
#       FLOAT accuracy | The total accuracy of the classification across all the generated images.
#       FLOAT precision | The total precision of the classification across all the generated images.
def getDatasetScore(gt_fol_dir, gen_img_dir, folds):
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

#Aliasing sensitivty and specificty into the truePosRate and trueNegativeRate respectively as these terms are quite often interchanged.
getSensitivity = getTruePosRate
getSpecificity = getTrueNegativeRate

#   reportCSV()
#   Walks through the all the datasets generated in the all-dataset folder in the BASE_DIR.
#   Computes all their scores, decomposes the folder name to recover all the parameters used and produces a .csv table to summarize all results.
#   CSV file is written on the BASE_DIR path during the process. But the function returns NONE.
#
#
#   <Input>
#       NONE
#   <Output>
#       NONE
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

                sensitivity, specificity, precision, accuracy = getDatasetScore(BASE_DIR + 'sourceCT/annotated_masks/',
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

#   *Private*
#   1.  Called By : ReportCSV()
#
#   _getPreProcessSettings(STRING dataset_name)
#   Recovers all the pre-processing parameters based on the dataset_name.
#
#
#   <Input>
#       required STRING dataset_name | A dataset_name, a.k.a. the folder storing the binary and preprocessed image directories.
#   <Output>
#       STRING gamma    |   The String value representing the gamma parameter used for the given dataset_name
#       STRING useHistEQ    | The String value representing the boolean decision of whether Histogram Equalization was used for the given dataset_name
#       STRING useCanny | The String value representing the boolean decision of whether Histogram CannyEdge was used for the given dataset_name
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

#   *Private*
#   1.  Called By : ReportCSV()
#
#   _getTrainingParams(STRING model_exc)
#   Recovers all the pre-processing parameters based on the dataset_name.
#
#
#   <Input>
#       required STRING model_exc | The folder name of the directory containing the output images of an SVM execution.
#   <Output>
#       String dims | The String value representing the sliding window dimensions of the SVM execution. Format of {X.size}x{Y.size}
#       STRING c    |   The String value representing the c parameter used in the Linear SVM.
#       STRING useSDV    | The String value representing the boolean decision of whether Std. Deviation Coefficient was used as feature for training.
#       STRING useCCM | The String value representing the boolean decision of whether Central Euclidean Distance was used as feature for training.
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

