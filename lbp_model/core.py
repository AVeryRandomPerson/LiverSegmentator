from imutils import paths

import os
import cv2
import numpy as np
import logs

def maskToAnnotation(src):
    annotation = Annotation(src)
    image = cv2.imread(src,0)
    for y in range(0,len(image)):
        for x in range(0, len(image[0])):
            if image[y][x] > 0:
                annotation.appendCoords((y,x))

    annotation.computeCenter()
    return annotation

# returns a whole list of key point objects.
def annotationFromMaskFolder(fol_path):
    all_annotations = []
    for img in paths.list_images(fol_path):
        annotation = maskToAnnotation(img)
        all_annotations = all_annotations + [annotation]

    return all_annotations

# Visualize kps for debug purposes
def visualizeKeyCoords(coords, canvas_size = (512, 512)):
    temp_img = np.zeros(canvas_size)
    for points in coords.coordinates:
        temp_img[points[0]][points[1]] = 255

    cv2.imshow("Key-points Acquired", temp_img)
    cv2.waitKey(0)

def writeAnnotation(annotation, out_path):
    annotation_file = open(out_path,'w+')

    annotation_file.write("{0}?".format(annotation.src.split('/').pop().split('.')[0]))    #Save The src name
    for coords in annotation.coordinates:
        annotation_file.writelines("{0}-{1} ".format(coords[0], coords[1]))

    #Append the center coordinate to the tail after all key points recorded.
    if(annotation.center == (0,0)): annotation.computeCenter()
    annotation_file.write("{0}-{1}".format(annotation.center[0], annotation.center[1]))
    annotation_file.close()


def writeAnnotationToFolder(annotations_list, out_dir):
    for annotation in annotations_list:
        out_path = out_dir+annotation.getName() + '.txt'
        writeAnnotation(annotation, out_path)


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


def readAnnotationFolder(in_dir):
    all_annotations = []
    for file in os.listdir(in_dir):
        if file.endswith('.txt'):
            annotation = readAnnotation(in_dir + file)
            all_annotations = all_annotations + [annotation]

    return all_annotations

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


def generateTextureFromList(annotation_list, src_dir, out_dir, dimensions=(73,73)):
    for annotation in annotation_list:
        generateTexture(annotation, src_dir, out_dir, dimensions)



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

#kps = annotationFromMaskFolder("C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/1x1-8r/liver/training/")
#writeAnnotationToFolder(kps, 'C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/annotation_coordinates/')
#all_kps = readAnnotationFolder('C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/annotation_coordinates/')

#kp = readAnnotation('C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/annotation_coordinates/scan8.txt')
#print(kp.src)
#generateTexture(kp, "C:/Users/acer/Desktop/TestSamples/ML-Dataset/CT_SCAN/training/" ,"C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP_Texture/73x73-annote/")


