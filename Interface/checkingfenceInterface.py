from mycode.oldcare.track import CentroidTracker
from mycode.oldcare.track import TrackableObject
import numpy as np
import imutils
import time
import dlib
import cv2
import os
from Interface import Communication

# 全局变量
prototxt_file_path = r'D:\PythonProject\oldCare-CV\models\mobilenet_ssd\MobileNetSSD_deploy.prototxt'
# Contains the Caffe deep learning model files.
# We’ll be using a MobileNet Single Shot Detector (SSD),
# “Single Shot Detectors for object detection”.
model_file_path = r'D:\PythonProject\oldCare-CV\models\mobilenet_ssd\MobileNetSSD_deploy.caffemodel'
output_fence_path = './supervision/fence'
skip_frames = 30  # of skip frames between detections
# your python path
python_path = r'D:\Anaconda3\envs\tensorflow\python.exe'
# 超参数
# minimum probability to filter weak detections
minimum_confidence = 0.80

# 物体识别模型能识别的物体（21种）
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]
trackers = []
trackableObjects = {}
net = cv2.dnn.readNetFromCaffe(prototxt_file_path, model_file_path)
totalFrames = 0
totalDown = 0
totalUp = 0


def checkingfence(grabbed, frame, net, fps):
    global totalFrames, totalDown, totalUp, trackers, trackableObjects
    if not grabbed:
        return
    W = None
    H = None
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    # trackers = []
    # trackableObjects = {}
    # 得到当前时间
    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                 time.localtime(time.time()))

    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > minimum_confidence:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # draw a rectangle around the people
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))
                    event_desc = '有人闯入禁止区域!!!'
                    event_location = '院子'
                    print('[EVENT] %s, 院子, 有人闯入禁止区域!!!' % (current_time))
                    cv2.imwrite(os.path.join(output_fence_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                frame)  # snapshot

                    Communication.insertevent("禁止区域入侵检测", event_desc, event_location, -1)
                    # insert into database
                    # command = '%s inserting.py --event_desc %s --event_type 4 --event_location %s' % (
                    # python_path, event_desc, event_location)
                    # p = subprocess.Popen(command, shell=True)

                # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4,
                   (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        # ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    totalFrames += 1
    fps.update()
    return frame

# def checkingfence(grabbed, frame, net, fps):

