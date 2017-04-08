import cv2
import numpy as np
from imutils import paths
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import os

from lbp_model import core
import logging
import logs

from lbp_model import lbp

prediction_log = logs.setupLogger("prediction_log",
                                  "C:/Users/acer/Desktop/TestSamples/Logs/Predicting/prediction.txt")
training_log = logs.setupLogger("training_log",
                                "C:/Users/acer/Desktop/TestSamples/Logs/Training/annotatedTraining.txt")

# Coordinate (X,Y)
def trainLBP(img, key_points, descriptor ,tile_dimensions = (5,5)):
    data = []
    labels = []

    tile_w = tile_dimensions[0]
    tile_h = tile_dimensions[1]

    if(tile_w < descriptor.radius or tile_h < descriptor.radius):
        training_log.info("Failed to train due to incompatible radius and dimension")
        return False

    buffered_img = np.zeros((len(img)+2*tile_h+1,len(img[0])+2*tile_w+1))
    buffered_img[tile_h:len(img)+tile_h,tile_w:len(img[0])+tile_w] = img

    for y in range(tile_h,len(img)+tile_h):
        training_log.info("Training completed at row {0}".format(y-tile_h))
        for x in range(tile_w,len(img)+tile_w):
            texture = buffered_img[y - tile_h:y + tile_h + 1, x - tile_w:x + tile_w + 1]
            hist = descriptor.describe(texture)
            data.append(hist)

            if(key_points and (y == key_points[0][0] + tile_h) and (x == key_points[0][1] + tile_w)):
                labels.append('liver')
                del key_points[0]
            else:
                labels.append('non-liver')

    training_log.info("COMPLETED TRAINING!")
    return data, labels

def trainLBPFolder(fol_dir, annotations_list, descriptor, tile_dimensions = (5,5)):
    all_data = []
    all_labels = []

    for annotation in annotations_list:
        img = cv2.imread(fol_dir + annotation.getName() + '.jpg',0)
        key_points = annotation.coordinates

        data, labels = trainLBP(img, key_points, descriptor, tile_dimensions)
        all_data = all_data + data
        all_labels = all_labels + labels

    return all_data, all_labels


def predictImageFolder(fol_dir, model, descriptor, tile_dimensions =(5,5)):
    for img in paths.list_images(fol_dir):
        mask = predictImage(cv2.imread(img, 0),
                     model,
                     descriptor,
                     tile_dimensions)

        cv2.imwrite("C:/Users/acer/Desktop/TestSamples/ML-Dataset/Bin-Results/lsvc/LSVC-C-100/" + img.split('/').pop(),mask)


# Refactor for custom path. SoftCode
def predictImage(img, model, descriptor ,tile_dimensions = (5,5)):

    tile_w = tile_dimensions[0]
    tile_h = tile_dimensions[1]

    if(tile_w < descriptor.radius or tile_h < descriptor.radius):
        prediction_log.info("Failed to predict due to incompatible radius and dimension")
        return False

    buffered_img = np.zeros((len(img)+2*tile_h+1,len(img[0])+2*tile_w+1))
    buffered_img[tile_h:len(img)+tile_h,tile_w:len(img[0])+tile_w] = img
    img_mask = np.zeros((len(img),len(img[0])))

    for y in range(tile_h,len(img)+tile_h):
        for x in range(tile_w,len(img)+tile_w):

            texture = buffered_img[y - tile_h:y + tile_h + 1, x - tile_w:x + tile_w + 1]
            hist = descriptor.describe(texture)
            data.append(hist)
            prediction = model.predict(hist.reshape(1, -1))
            if(prediction[0]=='liver'):
                img_mask[y-tile_h][x-tile_w] = 255

        if(y%50 == 0):
            prediction_log.info("Prediction completed at row {1}".format(y - tile_h))

    return img_mask

def fit_models(model_list, data, labels):
    for model in model_list:
        model.fit(data, labels)


def predictVisualize(lbp_radius, descriptor, model_list, model_names, src_test_dir, write_root, texture_width=36, texture_height=36):
    for image_path in paths.list_images(src_test_dir):
        temp_img = cv2.imread(image_path,0)
        img_bin = np.zeros((len(temp_img),len(temp_img[0])))
        img_data = np.zeros((len(temp_img) + 2*lbp_radius, len(temp_img[0]) + 2*lbp_radius))
        img_data[lbp_radius:len(temp_img)+lbp_radius, lbp_radius:len(temp_img[0])+lbp_radius] = cv2.imread(image_path, 0)

        for model_index in range(len(model_list)):
            for y in range(lbp_radius, len(temp_img)+lbp_radius):
                for x in range(lbp_radius, len(temp_img[0])+lbp_radius):
                    xmin = x + lbp_radius*0 - texture_width//2
                    xmax = x - lbp_radius*0 + texture_width//2
                    ymin = y + lbp_radius*0 - texture_height//2
                    ymax = y - lbp_radius*0 + texture_height//2
                    data = img_data[ymin:ymax, xmin:xmax]
                    hist = descriptor.describe(data)
                    prediction = model_list[model_index].predict(hist.reshape(1, -1))
                    if (prediction[0] == 'liver'):
                        img_bin[y - lbp_radius][x - lbp_radius] = 255

                logging.debug("{0} -- Scanning x-line : {1}".format(model_names[model_index], y - lbp_radius))
            logging.debug("{0} -- Done : {1}".format(model_names[model_index],image_path.split('/').pop()))
            write_dir = write_root + model_names[model_index] + '/' + image_path.split('/').pop()+ '.jpg'
            cv2.imwrite(write_dir, img_bin)


if __name__ == "__main__":
    TRAINING_LIVER = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/1x1-8r/liver/training/"
    TRAINING_NONLIVER = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/1x1-8r/non-liver/training/"
    TESTING_CT = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/CT_SCAN/testing/"
    BIN_RESULTS = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/Bin-Results/"

    kp = core.readAnnotation('C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/annotation_coordinates/scan8.txt')
    img = cv2.imread("C:/Users/acer/Desktop/TestSamples/ML-Dataset/CT_SCAN/training/scan8.jpg",0)
    desc = lbp.LocalBinaryPatterns(27, 10)
    data, labels = trainLBP(img,kp.coordinates,desc,tile_dimensions=(73,73))

    lsvc = LinearSVC(C=100, random_state=42)
    lsvc.fit(data, labels)
    predictImageFolder("C:/Users/acer/Desktop/TestSamples/ML-Dataset/CT_SCAN/testing/", lsvc, desc, tile_dimensions=(73, 73))



