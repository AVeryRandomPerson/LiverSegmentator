import os
import random
from imutils import paths


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


def getFiles(folder_dir):
    filelist = []
    for file in paths.list_images(folder_dir):
        img_name = file.split('/').pop()
        filelist.append(img_name)
    return filelist


def generateTrainIndex(total_files, k):
    train_sample_indexes = []
    partition_size = total_files//k
    for i in range(0,partition_size):
        index = (k * i) +  random.randint(0,(k-1))
        train_sample_indexes.append(index)


    return train_sample_indexes #.reverse()


if __name__ == '__main__':
    kcrossvalidate('C:/Users/acer/Desktop/TestSamples/ML-Dataset/CT_SCAN/lbp_reference/controlled/', 5)