import TextureFinder
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2

TRAINING = "C:/Users/acer/Desktop/TestSamples/LBP/liver/training/"
TRAINING2 = "C:/Users/acer/Desktop/TestSamples/LBP/non-liver/training/"
TESTING = "C:/Users/acer/Desktop/TestSamples/LBP/non-liver/testing/"

# construct the argument parse
# and then parse the arguments


#initializer the local binary patterns descriptor along with the data
#and label lists

desc = TextureFinder.LocalBinaryPatterns(24,8)
data = []
labels = []


for image_path in paths.list_images(TRAINING):
    img = cv2.imread(image_path,0)
    hist = desc.describe(img)


    labels.append(image_path.split("/")[-3])
    data.append(hist)

for image_path in paths.list_images(TRAINING2):
    img = cv2.imread(image_path,0)
    hist = desc.describe(img)


    labels.append(image_path.split("/")[-3])
    data.append(hist)



model = LinearSVC(C=100, random_state=42)
model.fit(data, labels)


for image_path in paths.list_images(TESTING):
    img = cv2.imread(image_path,0)
    hist = desc.describe(img)

    prediction = model.predict(hist.reshape(1,-1))
    print(prediction)
