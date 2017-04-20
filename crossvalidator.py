import os
import random
from imutils import paths


#   kcrossvalidate(STRING folder_dir, INT k, BOOLEAN split_to_folder)
#   performs kcrossvalidation on image files found in a given folder directory.
#   Either the image files will be partitioned into 2 separate folders OR 2 text files listing their group members is created.
#   This operation is done depending on the BOOLEAN split_to_folder. The function returns NONE.
#
#
#   <Input>
#       required STRING folder_dir | The directory path containing all image samples
#       optional INT k | The k value of k cross validation. The number of test samples generated will be equal to N // k
#       optional BOOLEAN split_to_folder | The decision of the images in the directory should be moved to separate folders of 'test' and 'train' respectively.
#                                          TRUE = File list not generated. Image files moved to separated folders.
#                                          FALSE = File list is generated into 2 text files. {k}folds_train_list.txt and {k}folds_test_list.txt. Image files are not moved.
#
#   <Output>
#       NONE
def kcrossvalidate(folder_dir, k = 10, split_to_folder = False):
    file_list = getFiles(folder_dir)
    try:

        train_sample_indexes = generateTrainIndex(len(file_list),k)
        i = 1

        if(split_to_folder):
            for file in file_list:
                if (len(train_sample_indexes) == 0):
                    os.rename(folder_dir + file, folder_dir + 'training/' + file)
                elif(i == train_sample_indexes[0]):
                    os.rename(folder_dir + file, folder_dir + 'testing/' + file)
                    train_sample_indexes.pop(0)
                else:
                    os.rename(folder_dir + file, folder_dir + 'training/' + file)

                i+=1

        else:
            write_dir = folder_dir.split('/')
            write_dir.pop()
            write_dir.pop()
            write_dir = '/'.join(write_dir)
            write_dir = write_dir + '/kfolds_list/'

            if not(os.path.exists(write_dir)):
                os.makedirs(write_dir + '{0}folds_test_list.txt'.format(k))
                os.makedirs(write_dir + '{0}folds_train_list.txt'.format(k))

            test_file = open(write_dir + '{0}folds_test_list.txt'.format(k), 'w+')
            train_file = open(write_dir + '{0}folds_train_list.txt'.format(k), 'w+')

            for file in file_list:
                if (len(train_sample_indexes) == 0):
                    train_file.writelines(file.split('.')[0] + '\n')
                elif (i == train_sample_indexes[0]):
                    test_file.writelines(file.split('.')[0] + '\n')
                    train_sample_indexes.pop(0)
                else:
                    train_file.writelines(file.split('.')[0] + '\n')

                i+=1

            test_file.close()
            train_file.close()


    except:

        raise Exception("No valid image files in given directory")


#   getFiles(STRING folder_dir)
#   gets a list of filenames without their format (like .jpg) in a given folder directory.
#
#
#   <Input>
#       required STRING folder_dir | The directory path containing image files.
#   <Output>
#       1D-ARRAY<String> filelist | A list of filenames without format tags.
def getFiles(folder_dir):
    filelist = []
    for file in paths.list_images(folder_dir):
        img_name = file.split('/').pop()
        filelist.append(img_name)
    return filelist


#   generateTrainIndex(INT total_files, INT k)
#   given the total number of files and the k value of the k-cross validation scheme, generates at random,
#   the index of files targetted to be test files. Index are distributed between each partition.
#
#
#   <Input>
#       required INT total_files | The total number of files.
#       required INT k | k value of the cross validation scheme.
#   <Output>
#       1D-ARRAY<Int> train_sample_indexes | The list of index of train files. (The values are indexed across the entire sample)
def generateTrainIndex(total_files, k):
    train_sample_indexes = []
    partition_size = total_files//k
    for i in range(0,partition_size):
        index = (k * i) +  random.randint(0,(k-1))
        train_sample_indexes.append(index)


    return train_sample_indexes

