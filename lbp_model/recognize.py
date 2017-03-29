import cv2
import numpy as np
from imutils import paths
from sklearn.svm import LinearSVC

from lbp_model import lbp

TRAINING_LIVER = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP/liver/training/"
TRAINING_NONLIVER = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP/non-liver/training/"
TESTING = "C:/Users/acer/Desktop/TestSamples/ML-Dataset/LBP/non-liver/testing/"

# construct the argument parse
# and then parse the arguments


#initializer the local binary patterns descriptor along with the data
#and label lists

desc = lbp.LocalBinaryPatterns(24, 8)
data = []
labels = []


for image_path in paths.list_images(TRAINING_LIVER):
    img = cv2.imread(image_path,0)
    hist = desc.describe(img)


    labels.append(image_path.split("/")[-3])
    data.append(hist)

for image_path in paths.list_images(TRAINING_NONLIVER):
    img = cv2.imread(image_path,0)
    hist = desc.describe(img)


    labels.append(image_path.split("/")[-3])
    data.append(hist)



model = LinearSVC(C=100, random_state=42)
model.fit(data, labels)


img_bin = np.zeros((396,504))
img_data = cv2.imread("C:/Users/acer/Desktop/TestSamples/BodyOnly/Mixed/I0000091.jpg",0)
for Y in range(0,11):
    y = 36 * Y
    for X in range (0,14):
        x = 36 * X

        data = img_data[y:y+36,x:x+36]
        hist = desc.describe(data)
        prediction = model.predict(hist.reshape(1,-1))
        print("y = {0} ; x = {1} ; dimensions = ({2} x {3}) ;prediction = {4}".format(y,x,len(data),len(data[0]),prediction))

        if(prediction[0] == 'liver'):
            img_bin[y:y+36,x:x+36] = 255


cv2.imwrite("C:/Users/acer/Desktop/TestSamples/BinI0000091.jpg",img_bin)