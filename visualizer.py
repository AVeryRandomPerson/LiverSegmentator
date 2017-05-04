

if __name__ == '__main__':
    import argparse
    from CONSTANTS import BASE_DIR
    import cv2
    from imutils import paths

    parser = argparse.ArgumentParser()
    parser.add_argument("--outputPath", help="The directory of output images of the SVM.")
    args = parser.parse_args()

    annotated_path = BASE_DIR + 'sourceCT/annotated_masks/'
    for img_path in paths.list_images(args.outputPath):
        img_name = img_path.split('\\').pop()
        annotated_image_path = annotated_path + img_name.split('.')[0] + '.png'

        annotated_img = cv2.imread(annotated_image_path,0)
        output_img = cv2.imread(img_path,0)

        score_img = cv2.bitwise_xor(output_img, annotated_img)
        _, score_img = cv2.threshold(score_img, 80, 255, cv2.THRESH_BINARY)

        cv2.imshow(img_name, score_img)
        cv2.waitKey(0)

