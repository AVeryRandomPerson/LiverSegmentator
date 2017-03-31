import os
import random

folder_path = "C:/Users/acer/Desktop/TestSamples/BodyOnly/Mixed/"

accepted_formats = ('.jpg', '.png')

def kcrossvalidate(folder_path, accepted_formats, k = 10):
    file_list = getFiles(folder_path, accepted_formats)
    try:

        train_sample_indexes = generateTrainIndex(len(file_list),k)
        i = 1
        for file in file_list:
            if (len(train_sample_indexes) == 0):
                #print(folder_path+'training/'+file)
                os.rename(folder_path+file, folder_path + 'training/' + file)
            elif(i == train_sample_indexes[0]):
                print(i, train_sample_indexes[0], folder_path + 'testing/' + file)
                os.rename(folder_path+file, folder_path + 'testing/' + file)
                train_sample_indexes.pop(0)
            else:
                #print(folder_path + 'training/' + file)
                os.rename(folder_path+file, folder_path + 'training/' + file)
            i+=1
    except:

        print("No valid image files in given directory")


def getFiles(folder_path, accepted_formats):
    filelist = []
    for file in os.listdir(folder_path):
        if(file.endswith(accepted_formats)):
            filelist.append(file)
    return filelist


def generateTrainIndex(total_files, k):
    train_sample_indexes = []
    partition_size = total_files//k
    for i in range(0,partition_size):
        index = (k * i) +  random.randint(0,(k-1))
        train_sample_indexes.append(index)

    return train_sample_indexes #.reverse()



if __name__  == '__main__':
    kcrossvalidate(folder_path,accepted_formats)
