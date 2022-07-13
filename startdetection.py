# -*- coding: utf-8 -*-
import argparse
import json
import subprocess

import cv2
import time
import imutils
import numpy as np
import threading as t

from imutils.video import FPS

from Interface.checkingInterface import checkingstrangersandfacialexpression, faceutil, facial_expression_model
from Interface.interactionInterface import checkingvolunteeractivity,faceutil
from Interface.falldetectionInterface import detect_fall
from Interface.checkingfenceInterface import checkingfence, net
import requests

# rtmpsrc1 = "http://zrp.cool:7001/live/movie.flv"
# rtmpsrc2 = "http://zrp.cool:7001/live/movie.flv"
# rtmpsrc3 = "http://zrp.cool:7001/live/movie.flv"
# rtmpsrc4 = "http://zrp.cool:7001/live/movie.flv"

# global frame1, frame2, frame3, frame4

# vs1 = cv2.VideoCapture(rtmpsrc1)
# vs2 = cv2.VideoCapture(rtmpsrc2)
# vs3 = cv2.VideoCapture(rtmpsrc3)
# vs4 = cv2.VideoCapture(rtmpsrc4)
# time.sleep(1)

# rtmp1 = "rtmp://zrp.cool:1935/live/L17LTlsVqMNTZyLKMIFSD2x28MlgPJ0SDZVHnHJPxMKi0tWx"
# rtmp2 = "rtmp://zrp.cool:1935/live/u3pQJ71N5GWfOIGTdSWXbRLGAwD1IkzuZ5G1pEDzqqm3sncC"
# rtmp3 = "rtmp://zrp.cool:1935/live/Yry01AuHiK7FDcCc35S4IzoOjgm2v8KyBpNlS52DyhMEXiJe"
# rtmp3 = "rtmp://zrp.cool:1935/live/v6e8bqQKRMnD4MR636KLtiuMzXX0NXjBvzYUgOKWKhDY2j42"

rtmp = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 推流参数
command = ['ffmpeg',
           '-y',
           '-re',
           # '-thread_queue_size', '512'
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', '640*480',  # 根据输入视频尺寸填写
           '-r', '25',
           '-i', '-',
           '-c:v', 'h264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv']

def getRTMP():
    r1 = requests.get('http://zrp.cool:8090/control/get?room=movie_01')
    r2 = requests.get('http://zrp.cool:8090/control/get?room=movie_02')
    r3 = requests.get('http://zrp.cool:8090/control/get?room=movie_03')
    r4 = requests.get('http://zrp.cool:8090/control/get?room=movie_04')

    key1 = json.loads(r1.text).get('data')
    key2 = json.loads(r2.text).get('data')
    key3 = json.loads(r3.text).get('data')
    key4 = json.loads(r4.text).get('data')

    rtmp1 = "rtmp://zrp.cool:1935/live/" + key1
    rtmp2 = "rtmp://zrp.cool:1935/live/" + key2
    rtmp3 = "rtmp://zrp.cool:1935/live/" + key3
    rtmp4 = "rtmp://zrp.cool:1935/live/" + key4

    rtmp.append(rtmp1)
    rtmp.append(rtmp2)
    rtmp.append(rtmp3)
    rtmp.append(rtmp4)

def checkstrangerthread():
    global command, rtmp, fourcc
    command1 = command
    command1.append(rtmp[0])
    print("11111")
    vs1 = cv2.VideoCapture(0)
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    size = (int(vs1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = vs1.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('./supervision/records/room_%s' % (time.strftime('%Y%m%d_%H%M%S')) + '.avi', fourcc, 3.0, size)

    while True:
        (grabbed, frame) = vs1.read()
        # cv2.imshow("origin", frame)
        out.write(frame)
        frame1 = checkingstrangersandfacialexpression(grabbed, frame, faceutil, facial_expression_model)
        img = cv2.resize(frame1, size)
        pipe.stdin.write(img.tobytes())
        cv2.imshow("1", frame1)

        k = cv2.waitKey(40) & 0xff
        if k == 27:
            break

    vs1.release()

    # 关闭窗口
    cv2.destroyAllWindows()

def checkvolunteerthread():
    global command, rtmp
    command2 = command
    command2.append(rtmp[1])
    print("22222")
    vs2 = cv2.VideoCapture(0)
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    size = (int(vs2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('./supervision/records/room_%s' % (time.strftime('%Y%m%d_%H%M%S')) + '.avi', fourcc, 3.0, size)
    while True:
        (grabbed, frame) = vs2.read()
        # cv2.imshow("origin", frame)
        out.write(frame)
        frame2 = checkingvolunteeractivity(grabbed, frame, faceutil)
        img = cv2.resize(frame2, size)
        pipe.stdin.write(img.tobytes())
        cv2.imshow("2", frame2)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    vs2.release()

    # 关闭窗口
    cv2.destroyAllWindows()


def checkfallthread():
    global command, rtmp
    command3 = command
    command3.append(rtmp[2])
    print("33333")
    vs3 = cv2.VideoCapture(0)
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    size = (int(vs3.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs3.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('./supervision/records/room_%s' % (time.strftime('%Y%m%d_%H%M%S')) + '.avi', fourcc, 3.0, size)
    while True:
        (grabbed, frame) = vs3.read()
        out.write(frame)
        # cv2.imshow("origin", frame)
        frame3 = detect_fall(grabbed, frame)
        img = cv2.resize(frame3, size)
        pipe.stdin.write(img.tobytes())
        cv2.imshow("3", frame3)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    vs3.release()

    # 关闭窗口
    cv2.destroyAllWindows()


def checkfencethread():
    global command, rtmp
    command4 = command
    command4.append(rtmp[3])
    print("44444")
    vs4 = cv2.VideoCapture(0)
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    size = (int(vs4.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs4.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('./supervision/records/room_%s' % (time.strftime('%Y%m%d_%H%M%S')) + '.avi', fourcc, 3.0, size)
    fps = FPS().start()
    while True:
        (grabbed, frame) = vs4.read()
        out.write(frame)
        # cv2.imshow("origin", frame)
        frame4 = checkingfence(grabbed, frame, net, fps)
        img = cv2.resize(frame4, size)
        pipe.stdin.write(img.tobytes())
        cv2.imshow("3", frame4)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    fps.stop()
    vs4.release()

    # 关闭窗口
    cv2.destroyAllWindows()

# def showframe():
#     while True:
#         ret,video = vs5.read()
#         cv2.imshow("oldcare_system", video)


if __name__ == '__main__':
    print('[INFO] 开始检测是否有人摔倒...')
    print('[INFO] 开始检测义工和老人是否有互动...')
    print('[INFO] 开始检测陌生人和表情...')
    print('[INFO] 开始检测禁止区域入侵...')
    getRTMP()
    t1 = t.Thread(target=checkstrangerthread)
    # t2 = t.Thread(target=checkvolunteerthread)
    # t3 = t.Thread(target=checkfallthread)
    # t4 = t.Thread(target=checkfencethread)

    t1.start()
    # t2.start()
    # t3.start()
    # t4.start()

    t1.join()
    # t2.join()
    # t3.join()
    # t4.join()


