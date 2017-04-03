import cv2
import numpy as np
from imutils import paths
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import os

import logging

from lbp_model import lbp

def train_LBP(fol_path, descriptor, texture_width=36, texture_height=36):
    data = []
    labels = []
    xmin = 0 + descriptor.radius
    ymin = 0 + descriptor.radius
    xmax = texture_width - descriptor.radius
    ymax = texture_height - descriptor.radius

    if (xmax <= 0 or ymax <= 0):
        logging.debug("TRAINING : Failed to train LBP. {0} ensure texture size and descriptor are compatible.".format(fol_path))
    else:
        for image_file in paths.list_images(fol_path):
            img = cv2.imread(image_file,0)
            img = img[ymin:ymax, xmin:xmax]
            hist = descriptor.describe(img)

            labels.append(fol_path.split("/")[-3])
            data.append(hist)

        logging.debug("TRAINING : Completed training LBP. Class : {0} - textures ({1}x{2})".format(labels[0],xmax-xmin,ymax-ymin))
    return data, labels

#Tiles
'''
for image_path in paths.list_images(TESTING_CT):
    img_bin = np.zeros((396, 504))
    img_data = np.zeros((436, 544))
    img_data[20:416, 20:524] = cv2.imread(image_path, 0)
    for Y in range(0,11):
        y = 36 * Y
        for X in range (0,14):
            x = 36 * X

            data = img_data[y +8 :y+36 -8,x +8 :x+36 -8]
            hist = desc.describe(data)
            prediction = model.predict(hist.reshape(1,-1))
            if(prediction[0] == 'liver'):
                img_bin[y:y+36,x:x+36] = 255

        cv2.imwrite(BIN_RESULTS + "/lsvc/tiles/" + 'bin' + image_path.split('/').pop() + '.jpg', img_bin)
        logging.debug("Scanning x-line : {0}".format(y-20))
    logging.debug("Done : {0}".format(image_path.split('/').pop()))
'''

'''
for image_path in paths.list_images(TESTING_CT):
    img_bin = np.zeros((396, 504))
    img_data = np.zeros((436, 544))
    img_data[20:416, 20:524] = cv2.imread(image_path, 0)
    for y in range(20,416):
        for x in range(20,524):
            data = img_data[y-10:y+10,x-10:x+10]
            hist = desc.describe(data)
            prediction = model.predict(hist.reshape(1,-1))
            if(prediction[0] == 'liver'):
                img_bin[y-20][x-20] = 255

        logging.debug("Scanning x-line : {0}".format(y-20))
    logging.debug("Done : {0}".format(image_path.split('/').pop()))
    cv2.imwrite(BIN_RESULTS + "/lsvc/" + 'bin' + image_path.split('/').pop() + '.jpg',img_bin)
'''

def fit_models(model_list, data, labels):
    for model in model_list:
        model.fit(data, labels)


def predictVisualize(lbp_radius, model_list, model_names, src_test_dir, write_root, texture_width=36, texture_height=36):
    for image_path in paths.list_images(src_test_dir):
        img_bin = np.zeros((396, 504))
        img_data = np.zeros((436, 544))
        img_data[20:416, 20:524] = cv2.imread(image_path, 0)

        for model_index in range(len(model_list)):
            for y in range(20, 416):
                for x in range(20, 524):
                    xmin = x + lbp_radius - texture_width//2
                    xmax = x - lbp_radius + texture_width//2
                    ymin = y + lbp_radius - texture_height//2
                    ymax = y - lbp_radius + texture_height//2
                    data = img_data[ymin:ymax, xmin:xmax]
                    hist = desc.describe(data)
                    prediction = model_list[model_index].predict(hist.reshape(1, -1))
                    if (prediction[0] == 'liver'):
                        img_bin[y - 20][x - 20] = 255

                logging.debug("{0} -- Scanning x-line : {1}".format(model_names[model_index], y - 20))
            logging.debug("{0} -- Done : {1}".format(model_names[model_index],image_path.split('/').pop()))
            write_dir = write_root + model_names[model_index] + '/' + image_path.split('/').pop()+ '.jpg'
            cv2.imwrite(write_dir, img_bin)

if __name__ == "__main__":
    TRAINING_LIVER = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/liver/training/"
    TRAINING_NONLIVER = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/non-liver/training/"
    TESTING_CT = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/CT_SCAN/testing/"
    BIN_RESULTS = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/Bin-Results/"

    logging.basicConfig(filename=BIN_RESULTS + 'training' + '_log.txt', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    desc = lbp.LocalBinaryPatterns(24, 8)


    liver_data, liver_labels = train_LBP(TRAINING_LIVER, desc)
    nonLiver_data, nonLiver_labels = train_LBP(TRAINING_NONLIVER, desc)
    data = liver_data + nonLiver_data
    labels = liver_labels + nonLiver_labels

    nu_model_sig2 = [NuSVC(nu=0.2, kernel='sigmoid', random_state=42)]
    nu_model_sig4 = [NuSVC(nu=0.4, kernel='sigmoid', random_state=42)]
    nu_model_sig6 = [NuSVC(nu=0.6, kernel='sigmoid', random_state=42)]
    nu_model_sig8 = [NuSVC(nu=0.8, kernel='sigmoid', random_state=42)]

    nu_model_rbf2 = [NuSVC(nu=0.2, random_state=42)]
    nu_model_rbf4 = [NuSVC(nu=0.4, random_state=42)]
    nu_model_rbf6 = [NuSVC(nu=0.6, random_state=42)]
    nu_model_rbf8 = [NuSVC(nu=0.8, random_state=42)]


    all_models = nu_model_sig2 + nu_model_sig4 + nu_model_sig6 + nu_model_sig8 + nu_model_rbf2 + nu_model_rbf4 + nu_model_rbf6 + nu_model_rbf8
    model_names = ["NU-MOD-SIG-0.2", "NU-MOD-SIG-0.4", "NU-MOD-SIG-0.6", "NU-MOD-SIG-0.8", "NU-MOD-RBF-0.2", "NU-MOD-RBF-0.4", "NU-MOD-RBF-0.6", "NU-MOD-RBF-0.8", ]
    for model_index in range(0,len(all_models)):
        all_models[model_index].fit(data, labels)
        logging.debug("Fitting data for model - {0} : DONE".format(model_names[model_index]))




    logging.basicConfig(filename=BIN_RESULTS + 'NU_SVC' + '_log.txt', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    predictVisualize(8,all_models,model_names,TESTING_CT,BIN_RESULTS+'nu-svc/')

'''
    linear_model_c0.1 = LinearSVC(C=0.1, weight = 

'''