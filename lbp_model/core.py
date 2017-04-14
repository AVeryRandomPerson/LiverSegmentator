from imutils import paths
from sklearn.svm import LinearSVC
from lbp_model import recognize

import pickle
import os
import cv2
import numpy as np
import logs

#   maskedToAnnotation(STRING path):
#   gets annotated coordinates of an annotated mask image.
#
#
#   <Input>
#      required STRING path | Source location of mask image.
#   <Output>
#      required ANNOTATION annotation | Stores annotated information of a mask image |See [Class Annotation]
def maskToAnnotation(path):
    annotation = Annotation(path)
    image = cv2.imread(path,0)
    for y in range(0,len(image)):
        for x in range(0, len(image[0])):
            if image[y][x] > 0:
                annotation.appendCoords((y,x))

    annotation.computeCenter()
    return annotation

#   annotationFromMaskFolder(STRING fol_path):
#   gets annotated coordinates of all annotated mask images within a folder directory.
#
#
#   <Input>
#      required STRING src | Source location of mask image(s) folder.
#   <Output>
#      required 1D-ARRAY Annotation | List of Annotation objects, each storing annotation information of an image. | See [Class Annotation]
def annotationFromMaskFolder(fol_path):
    all_annotations = []
    for img in paths.list_images(fol_path):
        annotation = maskToAnnotation(img)
        all_annotations = all_annotations + [annotation]

    return all_annotations

#   visualizeKeyCoords(1D-ARRAY<Tuple(int,int)> coordinates, TUPLE(int,int) canvas_size):
#   Draws a set of given coordinates onto a canvas as annotated pixels. Good for debugging purposes.
#   Image is displayed during process. But function will return NONE.
#
#
#   <Input>
#       required 1D-ARRAY<Tuple(int,int)> coordinates | A list of tuple of coordinates in (Y,X).
#       optional TUPLE(int,int) canvas_size | The dimensions of the canvas in (Y,X).
#   <Output>
#       NONE
def visualizeKeyCoords(coords, canvas_size = (512, 512)):
    temp_img = np.zeros(canvas_size)
    for points in coords:
        temp_img[points[0]][points[1]] = 255

    cv2.imshow("Key-points Acquired", temp_img)
    cv2.waitKey(0)

#   writeAnnotation(ANNOTATION annotation, STRING out_path):
#   given an Annotation object, write information to a .txt file.
#   file is created during process. But function will return NONE
#
#
#   <Input>
#       required ANNOTATION annotation | An annotation object, with stored annotation information.
#       optional STRING out_path | The output location. | Default = (512,512)
#   <Output>
#       NONE
def writeAnnotation(annotation, out_path):
    annotation_file = open(out_path,'w+')

    annotation_file.write("{0}?".format(annotation.src.split('/').pop().split('.')[0]))    #Save The src name
    for coords in annotation.coordinates:
        annotation_file.writelines("{0}-{1} ".format(coords[0], coords[1]))

    #Append the center coordinate to the tail after all key points recorded.
    if(annotation.center == (0,0)): annotation.computeCenter()
    annotation_file.write("{0}-{1}".format(annotation.center[0], annotation.center[1]))
    annotation_file.close()

#   writeAnnotationToFolder(1D-ARRAY<Annotation> annotations_list, String out_path):
#   given a list of Annotation object, and an output folder directory, write annotation(s) into the folder directory
#   file(s) are created during process. But function will return NONE
#
#
#   <Input>
#       required 1D-ARRAY<Annotation> annotations_list | A list of annotation object(s), with stored annotation information.
#       required STRING out_path | The output directory location. (The folder)
#   <Output>
#       NONE
def writeAnnotationToFolder(annotations_list, out_dir):
    for annotation in annotations_list:
        out_path = out_dir+annotation.getName() + '.txt'
        writeAnnotation(annotation, out_path)

#   readAnnotation(STRING path):
#   reads the annotation information from a .txt file and saves it into an Annotation object
#   see [Class Annotation]
#
#
#   <Input>
#       required STRING path | the location of the annotation .txt file.
#   <Output>
#       ANNOTATION annotation | the annotation object containing all the information from the .txt file.
def readAnnotation(path):
    annotation_file = open(path)

    annotation_data = annotation_file.read().split('?')     #Get The src name
    src_name = annotation_data[0]
    key_points = annotation_data[1].split(' ')
    center = key_points.pop()

    annotation = Annotation(src_name)
    for point in key_points:
        coords = point.split('-')
        coords = list(map(int, coords))
        annotation.appendCoords(tuple(coords))

    annotation.center = center
    annotation_file.close()
    return annotation

#   readAnnotationFolder(STRING in_dir):
#   reads the information of all annotation(s) from all .txt file(s) in a given folder directory
#   and saves them into a 1D-Array list of Annotation object(s).
#   see [Class Annotation]
#
#
#   <Input>
#       required STRING in_dir | the location of the folder containing annotation .txt file(s).
#       optional 1D-ARRAY<String> file_list | a list of targeted files to read. IF none specified, all .txt files will be taken.
#   <Output>
#       1D-ARRAY<Annotation> annotation | the 1D-Array containing all annotation(s) in the same order as the (.txt) files read.
def readAnnotationFolder(in_dir, file_list=None):
    all_annotations = []
    if file_list:
        for file in file_list:
            annotation = readAnnotation(in_dir + file + '.txt')
            all_annotations = all_annotations + [annotation]

    else:
        for file in os.listdir(in_dir):
            if file.endswith('.txt'):
                annotation = readAnnotation(in_dir + file)
                all_annotations = all_annotations + [annotation]

    return all_annotations

#   *OBSOLETE*
#   * No longer used because of poor optimization.
#
#   generateTexture(ANNOTATION annotation, STRING src_dir, STRING out_dir, TUPLE(int,int) dimensions):
#   from an Annotation object, captures an (X by Y) sized tile of the src image and saves it as a new image.
#   see [Class Annotation]
#   image file(s) are created during process. But function will return NONE
#
#
#   <Input>
#       required ANNOTATION annotation | the annotation object that stores annotated coordinates of an image.
#       required STRING src_dir | the location directory of the folder containing the image.
#       required STRING out_dir | the location directory of the folder where the cropped image is stored.
#       optional TUPLE(int,int) dimensions | the X by Y dimension of the texture window. | Default = (73,73)
#   <Output>
#       NONE
#
def generateTexture(annotation, src_dir, out_dir, dimensions=(73,73)):
    texture_log = logs.setupLogger("texture_log","C:/Users/acer/Desktop/TestSamples/Logs/TextureGeneration/" + annotation.getName() + "_log.txt")

    coords_list = annotation.coordinates
    temp_img = cv2.imread(src_dir + annotation.src + '.jpg', 0)
    width = dimensions [0]
    height = dimensions[1]
    if((height + width) %2 != 0):
        texture_log.info("Failed to generate textures. Invalid Dimensions provided. x and y must be both odd numbers")
        return False #Invalid dimension (Both MUST BE Odd Numbers)

    h2 = height//2
    w2 = width//2

    src_img = np.zeros((len(temp_img) + height+1 ,len(temp_img[0]) + width+1)) # +1 is a buffer
    src_img[h2:len(temp_img)+h2, w2:len(temp_img[0])+w2] = temp_img
    for y in range(h2,len(src_img) - h2 - 1):
        for x in range(w2, len(src_img) - w2 - 1):
            texture = src_img[y-h2:y+h2+1, x-w2:x+w2+1]
            if(coords_list and (y== coords_list[0][0]+h2) and (x== coords_list[0][1]+w2)):
                del coords_list[0]
                out_path = out_dir + '/liver/training/' + '{0}_{1}'.format(y - h2, x - w2) + annotation.src + '.jpg'
            else:
                out_path = out_dir + '/non-liver/training/' + '{0}_{1}'.format(y - h2, x - w2) + annotation.src + '.jpg'
            cv2.imwrite(out_path, texture)

        texture_log.info("Completed export texture for image at row : {0}".format(y-h2))
    texture_log.info("Textures successfully Generated.")

#   *OBSOLETE*
#   * No longer used because of poor optimization.
#
#   generateTexture(ANNOTATION annotation, STRING src_dir, STRING out_dir, TUPLE(int,int) dimensions):
#   from an Annotation object, captures an (X by Y) sized tile of the src image and saves it as a new image.
#   see [Class Annotation]
#   image file(s) are created during process. But function will return NONE
#
#
#   <Input>
#       required 1D-ARRAY<Annotation> annotation | the list of annotation object(s) that stores annotated coordinates of image(s).
#       required STRING src_dir | the location directory of the folder containing images.
#       required STRING out_dir | the location directory of the folder where the cropped images are stored.
#       optional TUPLE(int,int) dimensions | the X by Y dimension of the texture window. | Default = (73,73)
#   <Output>
#       1D-ARRAY<Annotation> annotation | the 1D-Array containing all annotation(s) in the same order as the (.txt) files read.
def generateTextureFromList(annotation_list, src_dir, out_dir, dimensions=(73,73)):
    for annotation in annotation_list:
        generateTexture(annotation, src_dir, out_dir, dimensions)

#   *ALIASED*
#   * Aliased functions perform the same as this function.
#   * >getTrainingList
#   * >getTestingList
#
#   getFileList(STRING path):
#   Acquires a list of file specified by a .txt file. Usually used to specify which filenames belong to test or train.
#
#   <Input>
#       required STRING src_dir | the location directory of the .txt file specifying a list of filename(s).
#   <Output>
#       1D-ARRAY<String> file_list | the list of filename(s) as specified in the .txt file given.
def getFileList(path):
    file_list = []
    with open(path) as f:
        file_list = f.read()
        file_list = file_list.split('\n')

    print(file_list)
    return file_list

getTrainingList = getFileList
getTestingList = getFileList


class Dataset():
    base_dir = ""
    dataset_dir = ""

    annotation_mask_source = ""
    annotation_source = ""
    clean_ct_source = ""

    processed_ct_dir = ""
    binary_dir = ""
    test_list = []
    train_list = []

    out_dir = ""
    def __init__(self, base_dir, lbp_descriptor, folds=5, sobel=False, gamma=None):
        self.descriptor = lbp_descriptor
        dataset_name = self._getDatasetName(lbp_descriptor, folds, sobel, gamma)
        self._generateDirectories(base_dir, lbp_descriptor, dataset_name)
        self.train_list = getTrainingList(self.base_dir + 'sourceCT/kfolds_list/{0}folds_train_list.txt'.format(folds))
        self.test_list = getTestingList(self.base_dir + 'sourceCT/kfolds_list/{0}folds_test_list.txt'.format(folds))
        self._preprocessSamples()

        print("Finished initializing")

    def _getDatasetName(self, lbp_descriptor, folds, sobel, gamma):
        dataset_name = '{0}folds_{1}n{2}r'.format(folds, lbp_descriptor.numPoints, lbp_descriptor.radius)
        if (sobel):
            dataset_name = dataset_name + '_sobel'

        if (gamma):
            dataset_name = dataset_name + '_gamma' + gamma

        dataset_name = dataset_name + '/'
        return dataset_name

    def _generateDirectories(self, base_dir, lbp_descriptor, dataset_name):
        self.base_dir = base_dir
        if(not self.base_dir.endswith("\\") or not self.base_dir.endswith("/")):
            base_dir = base_dir + '/'

        self.clean_ct_source = self.base_dir + 'sourceCT/' + 'CTs/'
        self.annotation_source = self.base_dir + 'sourceCT/' +'annotations/'
        self.annotation_mask_source = self.base_dir + 'sourceCT/' + 'annotation_mask_source/'

        self.dataset_dir = self.base_dir + 'all-datasets/' + dataset_name
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        self.processed_ct_dir = self.dataset_dir + 'preprocessed_samples/'
        self.binary_dir = self.dataset_dir + 'binaries/'
        self.out_dir = self.dataset_dir + 'output/'

    def _preprocessSamples(self):
        if not os.path.exists(self.processed_ct_dir):
            os.makedirs(self.processed_ct_dir)

        for img_path in paths.list_images(self.clean_ct_source):
            img = cv2.imread(img_path, 0)
            img_name = img_path.split('/').pop()
            lbp_img = self.descriptor.describe(img, mode='I')
            cv2.imwrite(self.processed_ct_dir + img_name, lbp_img)
            print("Exported - " + self.processed_ct_dir + img_name)

    def trainDataset(self, tile_dimensions=(73,73), useSDV=False, useCCostMeasure=False):
        annotations = readAnnotationFolder(self.annotation_source, self.train_list)

        final_bin_dir = self.binary_dir + 'lbp'
        if(useSDV): final_bin_dir = self.binary_dir + '_sdv'
        if(useCCostMeasure): final_bin_dir = self.binary_dir + '_ccm'
        final_bin_dir = final_bin_dir + '/'


        if not os.path.exists(final_bin_dir):
            os.makedirs(final_bin_dir)
        recognize.trainLBPFolder(self.processed_ct_dir,
                                 annotations,
                                 self.descriptor,
                                 tile_dimensions,
                                 final_bin_dir)

    def lsvcPredictData(self, tile_dimensions=(73,73), C=100.0, useSDV=False, useCCostMeasure=False):
        model_name = 'lbp'
        if(useSDV): model_name = model_name + '_sdv'
        if(useCCostMeasure): model_name = model_name + '_ccm'
        model_name = model_name + '/'

        data = []
        labels = []
        for file in self.train_list:
            with open(self.binary_dir + model_name + file + '.bin', 'rb') as f:
                    d, l = pickle.load(f)
                    data = data + d
                    labels = labels + l

        model = LinearSVC(C=C, random_state=42)
        model.fit(data, labels)

        model_name = 'c{0}_{1}x{2}_{3}'.format(C, tile_dimensions[0], tile_dimensions[1], model_name)
        final_out_dir = self.out_dir + model_name
        if not os.path.exists(final_out_dir):
            os.makedirs(final_out_dir)

        img_list = []
        for file_name in self.test_list:
            img = file_name + '.jpg'
            img_list = img_list + [img]
        recognize.predictImageFolder(self.processed_ct_dir,
                                     img_list,
                                     model,
                                     self.descriptor,
                                     final_out_dir,
                                     (73,73))

        del model



class Annotation():
    coordinates = []
    center = (0,0)

    def __init__(self, src):
        self.src = src

    def appendCoords(self,newCoord):
        self.coordinates = self.coordinates + [newCoord]

    def getName(self):
        return self.src.split('/').pop().split('.')[0]

    def computeCenter(self):
        Y = 0
        X = 0
        for (y,x) in self.coordinates:
            Y = Y + y
            X = X + x

        self.center = (Y//len(self.coordinates),  X//len(self.coordinates))


