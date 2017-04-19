import os
import cv2
import numpy as np

from imutils import paths

#   cropImagesInFolder(INT h1, INT h2, INT w1, INT w2, STRING fol_dir):
#   OVERWRITES all image(s) inside a folder directory with its cropped image derived
#   with a bounding frame of specified coordinate information.
#   images are OVERWRITTEN during the process. But the function returns None.
#
#
#   <Input>
#       required INT h1 | the highest pixel y-coordinate of the crop window. AKA the y value of the coordinate nearest to the top source image border.
#       required INT h2 | the lowest pixel y-coordinate of the crop window. AKA the y value of the coordinate nearest to the bottom source image border.
#       required INT w1 | the most left pixel x-coordinate of the crop window. AKA the x value of the coordinate nearest to the left source image border.
#       required INT w2 | the most right pixel x-coordinate of the crop window. AKA the x value of the coordinate nearest to the right source image border.
#       required STRING fol_dir | the folder directory which contains all the image(s).
#   <Output>
#       NONE
def cropImagesInFolder(h1, h2, w1, w2, fol_dir):
    for img_path in paths.list_images(fol_dir):
        img = cv2.imread(img_path)
        cv2.imwrite(img_path, img[h1:h2,w1:w2])

#   getHistogram(STRING fol_dir):
#   Computes a collective histogram of image intensity information of all image(s) in a specified folder directory.
#
#
#   <Input>
#       required STRING fol_dir | the folder directory which contains all the image(s).
#   <Output>
#       1D-ARRAY<int> final_hist | the histogram representing intensity population of all image(s) in a folder.
def getHistogram(fol_dir):
    final_hist = np.zeros((1,256))
    for file in os.listdir(fol_dir):
        if file.endswith(".jpg"):
            image = cv2.imread(fol_dir+file)

            cur_hist = cv2.calcHist([image],[0],None,[256],[0,256])
            final_hist = np.add(final_hist,cur_hist)

    return final_hist

#   thresholdImagesInFolder(STRING fol_dir, INT tresh_min, INT tresh_max):
#   Applies basic thresholding on image intensity.
#   Images are NOT overwritten. Instead, they are stored in a new folder 'thresheld/' on the specified directory.
#
#
#   <Input>
#       required STRING fol_dir | the folder directory which contains all the image(s).
#       required INT thresh_min | the minimum threshold intensity value.
#       required INT thresh_max | the maximum threshold intensity value.
#   <Output>
#       1D-ARRAY<int> final_hist | the histogram representing intensity population of all image(s) in a folder.
def thresholdImagesInFolder(fol_path, thresh_min, thresh_max):
    for img_path in paths.list_images(fol_path):
        image = cv2.imread(img_path)
        ret, thresh_im = cv2.threshold(image,
                                       thresh_min,
                                       thresh_max,
                                       cv2.THRESH_TOZERO)
        if not os.path.exists(fol_path + "threshed/"):
            os.makedirs(fol_path + "threshed/")
        cv2.imwrite(fol_path + "threshed/" + fol_path.split('/').pop(), thresh_im)

#   binaryAND(STRING src1_dir, STRING src2_dir, STRING out_path):
#   Combines two binary images together with a binary AND.
#   The process writes out binary images. But the function returns NONE.
#
#
#   <Input>
#       required STRING src1_dir | the folder directory which contains all the binary image(s).
#       required STRING src2_dir | the folder directory which contains all the binary image(s). Paired with src1_dir
#       required STRING out_path | the folder directory to output the final image(s).
#   <Output>
#       NONE
def binaryAND(src1_dir, src2_dir, out_dir):
    fol1_img = []
    fol2_img = []

    for img_path in paths.list_images(src1_dir):
        fol1_img.append(img_path)

    for img_path in paths.list_images(src2_dir):
        fol2_img.append(img_path)

    for i in range(0,len(fol2_img)):
        img1 = cv2.imread(fol1_img[i],0)
        img2 = cv2.imread(fol2_img[i],0)

        final_img = cv2.bitwise_and(img1,img2)
        cv2.imwrite(out_dir + fol1_img[i].split('/').pop(), final_img)

#   applySobel(STRING img_path, INT xOrd, INT yOrd, INT kSize):
#   Applies a specified sobel filter to the image.
#
#
#   <Input>
#       required NUMPY_ARRAY<Image> img | the numpy array representation of an imgge.
#       optional INT xOrd | the order of x. This is the sensitivity of the filter on the horizontal plane.
#       optional INT yOrd | the order of y. This is the sensitivity of the filter on the vertical plane.
#       optional INT kSize | the size of the filter. MUST be Odd number less than 31.
#   <Output>
#       1D-ARRAY<int> final_hist | the histogram representing intensity population of all image(s) in a folder.
def applyCanny(img, tresh1=100, tresh2=200, kSize=3):
    # process it as 64f then take abs value and fit it to 8u. This helps sobel filter around low intensity.
    if(kSize%2 == 1):

        cannyImg = cv2.Canny(img,tresh1,tresh2,apertureSize=kSize)
        cannyImg = cv2.add(cannyImg,img)

    else:
        raise Exception('please specify k as an odd number below 31')

    return cannyImg

#   gammaContrast(NUMPY_ARRAY[Image] img, FLOAT gamma):
#   Applies a specified sobel filter to the image.
#
#
#   <Input>
#       required NUMPY_ARRAY[Image] img | the numpy array representation of an imgge.
#       required FLOAT gamma | the gamma value for contrast correction.
#   <Output>
#       NUMPY_ARRAY[Image] /Anonymous/ | the gamma contrast corrected image.
def gammaContrast(img, gamma):
    invGamma = 1.0/gamma
    #Using a table to pass to cv2 LUT for max efficiency.
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    del invGamma
    return cv2.LUT(img, table)
