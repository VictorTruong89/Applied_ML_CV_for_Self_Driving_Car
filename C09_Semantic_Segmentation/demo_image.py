import argparse
import cv2
import os
import imutils
import time
import numpy as np


start = time.time()
SET_WIDTH = int(600)

normalize_image = 1 / 255.0
resize_image_shape = (1024, 512)

sample_img = cv2.imread('./images/example_02.jpg')
sample_img = imutils.resize(sample_img, width=SET_WIDTH)
img_height = sample_img.shape[0]
img_width = sample_img.shape[1]
# cv2.imshow('sample image', sample_img)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()

blob_img = cv2.dnn.blobFromImage(sample_img, normalize_image, resize_image_shape, 0,swapRB=True, crop=False)
# print(blob_img)

print("[INFO] Loading model...")
# PROJECT_DIR = "D:/CODE/NEURAL_NETWORK/Retrain_E_Net_TF"
PROJECT_DIR = "C:/Users/RRIS/Desktop/Applied_DL_CV_SDC/TUTORIALS/09_Implement_Segmentation"
cityscapes_path = os.path.join(PROJECT_DIR, "dataset", "cityscapes")
network_path = os.path.join(cityscapes_path, "enet-model.net")
label_path = os.path.join(cityscapes_path, "enet-classes.txt")
color_path = os.path.join(cityscapes_path, "enet-colors.txt")

cv_enet_model = cv2.dnn.readNet(network_path)
cv_enet_model.setInput(blob_img)
cv_enet_model_output = cv_enet_model.forward()

label_values = open(label_path).read().strip().split("\n")

IMG_OUTPUT_SHAPE_START = 1
IMG_OUTPUT_SHAPE_END = 4

(class_num, h, w) = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]
class_map = np.argmax(cv_enet_model_output[0], axis=0)

if os.path.isfile(color_path):
    CV_ENET_SHAPE_IMG_COLORS = open(color_path).read().strip().split("\n")
    CV_ENET_SHAPE_IMG_COLORS = [np.array(c.split(",")).astype("int") for c in CV_ENET_SHAPE_IMG_COLORS]
    CV_ENET_SHAPE_IMG_COLORS = np.array(CV_ENET_SHAPE_IMG_COLORS, dtype="uint8")
else:
    np.random.seed(42)
    CV_ENET_SHAPE_IMG_COLORS = np.random.randint(0, 255, size=(len(label_values) - 1, 3), dtype="uint8")
    CV_ENET_SHAPE_IMG_COLORS = np.vstack([[0, 0, 0], CV_ENET_SHAPE_IMG_COLORS]).astype("uint8")

mask_class_map = CV_ENET_SHAPE_IMG_COLORS[class_map]
mask_class_map = cv2.resize(mask_class_map, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
class_map = cv2.resize(mask_class_map, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

cv_enet_model_output = ((0.4 * sample_img) + (0.6 * mask_class_map)).astype("uint8")

my_legend = np.zeros( ( (len(label_values) * 25) + 25, 300, 3 ), dtype="uint8" )

for (i, (class_name, img_color)) in enumerate( zip(label_values, CV_ENET_SHAPE_IMG_COLORS) ):
    # Draw the class name & legends on the legends
    color_info = [int(color) for color in img_color]
    cv2.putText( my_legend, class_name, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
    cv2.rectangle( my_legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color_info), -1 )

end = time.time()

cv2.imshow("My_Legend", my_legend)
cv2.imshow("Img_Input", sample_img)
cv2.imshow("CV_Model_Output", cv_enet_model_output)
cv2.waitKey(1000)
cv2.destroyAllWindows()


print("[INFO] inference took {:.4f} seconds".format(end - start))