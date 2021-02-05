import os
import time
import cv2
import imutils
import numpy as np


DEFAULT_FRAME = 1
SET_WIDTH = 600

# PROJECT_DIR = "D:/CODE/NEURAL_NETWORK/Retrain_E_Net_TF"
PROJECT_DIR = "C:/Users/RRIS/Desktop/Applied_DL_CV_SDC/TUTORIALS/09_Implement_Segmentation"
cityscapes_path = os.path.join(PROJECT_DIR, "dataset", "cityscapes")
network_path = os.path.join(cityscapes_path, "enet-model.net")
label_path = os.path.join(cityscapes_path, "enet-classes.txt")
color_path = os.path.join(cityscapes_path, "enet-colors.txt")

class_labels = open(label_path).read().strip().split("\n")

if os.path.isfile(color_path):
    CV_ENET_SHAPE_IMG_COLORS = open(color_path).read().strip().split("\n")
    CV_ENET_SHAPE_IMG_COLORS = [np.array(c.split(",")).astype("int") for c in CV_ENET_SHAPE_IMG_COLORS]
    CV_ENET_SHAPE_IMG_COLORS = np.array(CV_ENET_SHAPE_IMG_COLORS, dtype="uint8")
else:
    
    np.random.seed(42)
    CV_ENET_SHAPE_IMG_COLORS = np.random.randint(0, 255, size=(len(class_labels) - 1, 3),
                               dtype="uint8")
    CV_ENET_SHAPE_IMG_COLORS = np.vstack([[0, 0, 0], CV_ENET_SHAPE_IMG_COLORS]).astype("uint8")

print("[INFO] loading model...")
cv_enet_model = cv2.dnn.readNet(network_path)

sv = cv2.VideoCapture('.//video//video.mp4')
sample_video_writer = None

print(sv)

prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
total = int(sv.get(prop))
print(total)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(sv.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    total = -1
#sample_video
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = sv.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # construct a blob from the frame and perform a forward pass
    # using the segmentation model
    normalize_image = 1 / 255.0
    resize_image_shape = (1024, 512)
    video_frame = imutils.resize(frame, width=SET_WIDTH)
    blob_img = cv2.dnn.blobFromImage(frame,  normalize_image,resize_image_shape, 0,
                                 swapRB=True, crop=False)
    cv_enet_model.setInput(blob_img)
    start = time.time()
    cv_enet_model_output = cv_enet_model.forward()
    end = time.time()

    # infer the total number of classes along with the spatial
    # dimensions of the mask image via the shape of the output array
    (Classes_num, height, width) = cv_enet_model_output.shape[1:4]

    # our output class ID map will be num_classes x height x width in
    # size, so we take the argmax to find the class label with the
    # largest probability for each and every (x, y)-coordinate in the
    # image
    classMap = np.argmax(cv_enet_model_output[0], axis=0)

    # given the class ID map, we can map each of the class IDs to its
    # corresponding color
    
    mask_class_map = CV_ENET_SHAPE_IMG_COLORS[classMap]

    # resize the mask such that its dimensions match the original size
    # of the input frame
    
    
    mask_class_map = cv2.resize(mask_class_map, (video_frame.shape[1], video_frame.shape[0]),
                      interpolation=cv2.INTER_NEAREST)

    # perform a weighted combination of the input frame with the mask
    # to form an output visualization
    
    
    cv_enet_model_output = ((0.3 * video_frame) + (0.7 * mask_class_map)).astype("uint8")

    # check if the video writer is None
    if sample_video_writer is None:
        print("sample_video_writer is None")
        # initialize our video writer
        fourcc_obj = cv2.VideoWriter_fourcc(*"MJPG")

        sample_video_writer = cv2.VideoWriter('./output/output_toronoto.avi', fourcc_obj, 30,
                                 (cv_enet_model_output.shape[1], cv_enet_model_output.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            
            execution_time = (end - start)
            print("[INFO] single video_frame took {:.4f} seconds".format(execution_time))

            print("[INFO] estimated total_time time: {:.4f}".format(
                execution_time * total))

    # write the output frame to disk
    
    sample_video_writer.write(cv_enet_model_output)

    # check to see if we should display the output frame to our screen
    if DEFAULT_FRAME > 0:
        cv2.imshow("Video Frame", cv_enet_model_output)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break

print("[INFO] cleaning up...")
sample_video_writer.release()
sv.release()