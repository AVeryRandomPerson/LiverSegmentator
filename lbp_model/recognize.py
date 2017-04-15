import cv2
import numpy as np
from imutils import paths
import pickle
import logs

prediction_log = logs.setupLogger("prediction_log",
                                  "C:/Users/acer/Desktop/TestSamples/Logs/Predicting/prediction.txt")
training_log = logs.setupLogger("training_log",
                                "C:/Users/acer/Desktop/TestSamples/Logs/Training/annotatedTraining.txt")

#   trainLBP(NUMPY_ARRAY[Image] img, 1D-ARRAY<Tuple(Int,Int)> key_points, LocalBinaryPatterns descriptor, TUPLE(int,int) tile_dimensions):
#   trains an image using LBP feature using a tile size (X , Y) sliding window. Image(s) must already be preprocessed.
#
#
#   <Input>
#       required NUMPY_ARRAY[Image] img | A Grayscaled 2D-NUMPY_ARRAY representation of an image.
#       required 1D-ARRAY<Tuple(Int,Int)> key_points | The annotated key points of the given image.
#       required LBP descriptor | The descriptor object which generates lbp features. See [Class LocalBinaryPatterns]
#       optional TUPLE(int,int) tile_dimensions | The input tile size of the sliding window. Format is (X , Y)
#   <Output>
#       1D-ARRAY<Int> data | the histogram representing the lbp feature of the sliding window tile. Index is aligned with labels.
#       1D-ARRAY<String> labels | the string information of the class. | The list of classnames which is represented by each of the returned 1D-Array<Int> data. Index is aligned with data.
def trainLBP(img, key_points, descriptor, tile_dimensions = (5,5)):
    data = []
    labels = []

    xMin = max([tile_dimensions[0], descriptor.radius])
    yMin = max([tile_dimensions[1], descriptor.radius])

    buffered_img = np.zeros((len(img)+2*yMin+1,len(img[0])+2*xMin+1))
    buffered_img[yMin:len(img)+yMin,xMin:len(img[0])+xMin] = img

    for y in range(yMin, len(img) + yMin):
        for x in range(xMin, len(img) + xMin):
            texture = buffered_img[y - yMin:y + yMin + 1, x - xMin:x + xMin + 1]
            hist = descriptor.computeHistogram(texture)
            data.append(hist)

            if (key_points and (y == key_points[0][0] + yMin) and (x == key_points[0][1] + xMin)):
                labels.append('liver')
                del key_points[0]
            else:
                labels.append('non-liver')

        if (y % 50 == 0):
            training_log.info("Training completed at row {0}".format(y - yMin))

    training_log.info("COMPLETED TRAINING!")
    return data, labels

#   * OBSOLETE *
#   * No longer used due to optimization issues, and poor edging responses.
#
#   trainLBPWithTiles(NUMPY_ARRAY[Image] img, 1D-ARRAY<Tuple(Int,Int)> key_points, LocalBinaryPatterns descriptor, TUPLE(int,int) tile_dimensions):
#   trains an texture sample image using LBP feature using a tile size (X , Y). Note implementation is not sliding window.
#
#
#   <Input>
#       required NUMPY_ARRAY[Image] img | A Grayscaled 2D-NUMPY_ARRAY representation of an image.
#       required 1D-ARRAY<Tuple(Int,Int)> key_points | The annotated key points of the given image.
#       required LBP descriptor | The descriptor object which generates lbp features. See [Class Local LocalBinaryPatterns Pattern]
#       optional TUPLE(int,int) tile_dimensions | The input tile size of the sliding window. Format is (X , Y)
#   <Output>
#       1D-ARRAY<Int> data | the histogram representing the lbp feature of the sliding window tile. Index is aligned with labels.
#       1D-ARRAY<String> labels | the string information of the class. | The list of classnames which is represented by each of the returned 1D-Array<Int> data. Index is aligned with data.
def trainLBPWithTiles(img, key_points, descriptor ,tile_dimensions = (5,5)):
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
        for x in range(tile_w,len(img)+tile_w):
            texture = buffered_img[y - tile_h:y + tile_h + 1, x - tile_w:x + tile_w + 1]
            hist = descriptor.describe(texture, "H")
            data.append(hist)

            if(key_points and (y == key_points[0][0] + tile_h) and (x == key_points[0][1] + tile_w)):
                labels.append('liver')
                del key_points[0]
            else:
                labels.append('non-liver')

        if(y%50 ==0):
            training_log.info("Training completed at row {0}".format(y - tile_h))

    training_log.info("COMPLETED TRAINING!")
    return data, labels

#   trainLBP(STRING fol_dir, 1D-ARRAY<Annotation> annotations_list, LOCAL_BINARY_PATTERN descriptor, TUPLE(int,int) tile_dimensions, STRING bin_dir):
#   trains image(s) in a given directory using LBP feature using a tile size (X , Y) sliding window. Image(s) must already be preprocessed.
#
#
#   <Input>
#       required STRING fol_dir | The directory path containing all pre-processed images.
#       required 1D-ARRAY<Annotation> annotations_list | A list of Annotation objects corresponding to the images in the folder directory. See [Class Annotation]
#       required LBP descriptor | The descriptor object which generates lbp features. See [Class Local LocalBinaryPatterns Pattern]
#       optional TUPLE(int,int) tile_dimensions | The input tile size of the sliding window. Format is (X , Y)
#       optional STRING bin_dir | The output path of the binary directory. Output is written to binary files if specified, otherwise, function returns the entire list of data and labels
#   <Output>
#       bin_dir = None
#           1D-ARRAY<Int> all_data | the histogram representing the lbp feature of the sliding window tile. Index is aligned with labels.
#           1D-ARRAY<String> all_labels | the string information of the class. | The list of classnames which is represented by each of the returned 1D-Array<Int> data. Index is aligned with data.
#
#       bin_dir = valid path.
#           NONE
def trainLBPFolder(fol_dir, annotations_list, descriptor, tile_dimensions = (5,5), bin_dir = None):
    all_data = []
    all_labels = []

    for annotation in annotations_list:
        img = cv2.imread(fol_dir + annotation.getName() + '.jpg',0)
        key_points = annotation.coordinates

        data, labels = trainLBP(img, key_points, descriptor, tile_dimensions)


        if(not bin_dir):
            all_data = all_data + data
            all_labels = all_labels + labels
        else:
            with open(bin_dir + annotation.getName() + '.bin' ,'wb+') as f:
                pickle.dump([data,labels], f)


    return all_data, all_labels

#   predictImage(NUMPY_ARRAY[Image] img, LSVC model, LocalBinaryPatterns descriptor, TUPLE(int,int) tile_dimensions):
#   Using a sliding window of a specifiable size, predicts the class of every pixel to generate a binary image.
#
#
#   <Input>
#       required NUMPY_ARRAY[Image] img | A Grayscaled 2D-NUMPY_ARRAY representation of an image
#       required LSVC model | A LinearSVC model with training data already fit.
#       required LBP descriptor | The descriptor object which generates lbp features. See [Class LocalBinaryPatterns]
#       optional TUPLE(int,int) tile_dimensions | The input tile size of the sliding window. Format is (X , Y)
#   <Output>
#       NUMPY_ARRAY[Image] img | A Binary 2D-NUMPY_ARRAY representing the predicted image.
def predictImage(img, model, descriptor ,tile_dimensions = (5,5)):
    xMin = max([tile_dimensions[0], descriptor.radius])
    yMin = max([tile_dimensions[1], descriptor.radius])

    buffered_img = np.zeros((len(img)+2*yMin+1,len(img[0])+2*xMin+1))
    buffered_img[yMin:len(img)+yMin,xMin:len(img[0])+xMin] = img
    img_mask = np.zeros((len(img),len(img[0])))

    for y in range(yMin, len(img) + yMin):
        for x in range(xMin, len(img) + xMin):
            texture = buffered_img[y - yMin:y + yMin + 1, x - xMin:x + xMin + 1]
            hist = descriptor.computeHistogram(texture)
            prediction = model.predict(hist.reshape(1, -1))
            if (prediction[0] == 'liver'):
                img_mask[y - yMin][x - xMin] = 255

        if (y % 50 == 0):
            prediction_log.info("Prediction completed at row{0}".format(y - yMin))


    return img_mask

#   predictImageFolder(STRING fol_dir, 1D-ARRAY<String> img_list, LSVC model, LBP descriptor, STRING out_dir TUPLE(int,int) tile_dimensions):
#   Using a sliding window of a specifiable size, predicts the class of every pixel to generate a binary image.
#   Binary image files are generated and written to out_dir during process. But the function returns NONE.
#
#
#   <Input>
#       required STRING fol_dir | The directory of pre-processed images to predict.
#       required 1D-ARRAY<String> img_list | The list of image filenames specifying which images to read in the given directory.
#       required LSVC model | A LinearSVC model with training data already fit.
#       required LBP descriptor | The descriptor object which generates lbp features. See [Class LocalBinaryPatterns]
#       required STRING out_dir | The output directory storing the binary image.
#       optional TUPLE(int,int) tile_dimensions | The input tile size of the sliding window. Format is (X , Y)
#   <Output>
#       NONE
def predictImageFolder(fol_dir, img_list, model, descriptor, out_dir, tile_dimensions =(5,5)):
    if(img_list):
        for img in img_list:
            mask = predictImage(cv2.imread(fol_dir + img, 0),
                                model,
                                descriptor,
                                tile_dimensions)
            cv2.imwrite(out_dir + img,mask)

    else:
        for img in paths.list_images(fol_dir):
            mask = predictImage(cv2.imread(img, 0),
                         model,
                         descriptor,
                         tile_dimensions)

            cv2.imwrite(out_dir + img.split('/').pop(),mask)

#   fit_models(1D-ARRAY<LSVC> model_list, 1D-ARRAY<int> data, 1D-ARRAY labels):
#   Given data and labels, fit into all the models in a model list.
#   Each specified model will have data fit into it during the process. They can be used after this. But this function returns NONE.
#   Due to memory constrains it is unlikely this function will be used for big datasets.
#
#
#   <Input>
#       required 1D-ARRAY<LSVC> model_list | A list of all LSVC models.
#       required 1D-ARRAY<int> data | Array containing all the data values corresponding to labels
#       required 1D-ARRAY<int> labels | Array containing all the labels corresponding to data.

#   <Output>
#       NONE
def fit_models(model_list, data, labels):
    for model in model_list:
        model.fit(data, labels)
