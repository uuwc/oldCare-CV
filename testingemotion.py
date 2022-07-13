
from keras.models import load_model
import pandas as pd
import numpy as np
from tqdm import tqdm  # 进度条
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from mycode.oldcare.facial import FaceUtil
import numpy as np
import imutils
import cv2
import time
import argparse

# 传入参数
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=False, default='',
                help="")
args = vars(ap.parse_args())

# 全局变量
model_path = './models/face(5).h5'
input_video = args['filename']

# 全局常量
FACIAL_EXPRESSION_TARGET_WIDTH = 48
FACIAL_EXPRESSION_TARGET_HEIGHT = 48

# load the face detector cascade and emotion detector CNN
model = load_model(model_path)

# if a video path was not supplied, grab the reference to the webcam
if not input_video:
    camera = cv2.VideoCapture(0)
    time.sleep(2)
else:
    camera = cv2.VideoCapture(input_video)

faceutil = FaceUtil()

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame, then we
    # have reached the end of the video
    if input_video and not grabbed:
        break

    if not input_video:
        frame = cv2.flip(frame, 1)

    # resize the frame, convert it to grayscale, and then clone the
    # original frame so we can draw on it later in the program
    frame = imutils.resize(frame, width=600)

    face_location_list = faceutil.get_face_location(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # loop over the face bounding boxes
    for (left, top, right, bottom) in face_location_list:
        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN

        #roi = gray[top:bottom, left:right]

        roi = frame[top:bottom, left:right]
        roi = cv2.resize(roi, (FACIAL_EXPRESSION_TARGET_WIDTH,
                               FACIAL_EXPRESSION_TARGET_HEIGHT))

        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)

        #roi = roi.reshape(48, 48, 3)

        roi = np.expand_dims(roi, axis=0)
        roi = np.vstack([roi])




        # determine facial expression
        classes = model.predict(roi, batch_size=10)
        #classes = model.predict(roi)[0]
        num = np.argmax(classes)



        if num == 0:  # {'angry': 0, 'disgusted': 1, 'fearful': 1, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprised': 6}
            label = 'angry'
        if num == 1:
            label = 'disgusted'
        if num == 2:
            label = 'fearful'
        if num == 3:
            label = 'happy'
        if num == 4:
            label = 'neutral'
        if num == 5:
            label = 'sad'
        if num == 6:
            label = 'surprised'


        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 0, 255), 2)

    # show our detected faces along with smiling/not smiling labels
    cv2.imshow("Facial Expression Detect", frame)

    # Press 'ESC' for exiting video
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
