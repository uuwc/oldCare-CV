import cv2

# subprocess 模块允许我们启动一个新进程，并连接到它们的输入/输出/错误管道，从而获取返回值。
import subprocess


# 视频读取对象
# cap = cv2.VideoCapture(r"D:\PythonProject\oldCare-CV\tests\room_04.avi")
# cap = cv2.VideoCapture(0)
# # 读取一帧
# ret, frame = cap.read()

# 推流地址
# rtmp = "rtmp://123.56.148.46:1935/live/rfBd56ti2SMtYvSgD5xAV0YU99zampta7Z7S575KLkIZ9PYk"
# rtmp = "rtmp://zrp.cool:1935/live/rfBd56ti2SMtYvSgD5xAV0YU99zampta7Z7S575KLkIZ9PYk"
# # 推流参数
# command = ['ffmpeg',
#            '-y',
#            #'-re',
#            # '-thread_queue_size', '512'
#            '-f', 'rawvideo',
#            '-vcodec', 'rawvideo',
#            '-pix_fmt', 'bgr24',
#            '-s', '640*480',  # 根据输入视频尺寸填写
#            '-r', '25',
#            '-i', '-',
#            '-c:v', 'h264',
#            '-pix_fmt', 'yuv420p',
#            '-preset', 'ultrafast',
#            '-f', 'flv',
#            rtmp]

# # 创建、管理子进程
# pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 循环读取
# while cap.isOpened():
#     # 读取一帧
#     ret, frame = cap.read()
#     if frame is None:
#         print('read frame err!')
#         continue
#
#     # 显示一帧
#     cv2.imshow("frame", frame)
#
#     # 按键退出
#     k = cv2.waitKey(1) & 0xff
#     if k == 27:
#         break
#
#     # 读取尺寸、推流
#     img = cv2.resize(frame, size)
#     pipe.stdin.write(img.tobytes())
#
# # 关闭窗口
# cv2.destroyAllWindows()
#
# # 停止读取
# cap.release()

def pushstream(vs, frame):
    rtmp = "rtmp://zrp.cool:1935/live/rfBd56ti2SMtYvSgD5xAV0YU99zampta7Z7S575KLkIZ9PYk"
    # 推流参数
    command = ['ffmpeg',
               '-y',
               # '-re',
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
               '-f', 'flv',
               rtmp]

    # 创建、管理子进程
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    size = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if frame is None:
        print('read frame err!')

    # 读取尺寸、推流
    img = cv2.resize(frame, size)
    pipe.stdin.write(img.tobytes())
    print("推流成功")

if __name__ == '__main__':
    # cap = cv2.VideoCapture(r"D:\PythonProject\oldCare-CV\tests\room_04.avi")
    vs = cv2.VideoCapture(0)
    rtmp = "rtmp://zrp.cool:1935/live/rfBd56ti2SMtYvSgD5xAV0YU99zampta7Z7S575KLkIZ9PYk"
    # 推流参数
    command = ['ffmpeg',
               '-y',
               # '-re',
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
               '-f', 'flv',
               rtmp]
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    size = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while vs.isOpened():
        # 读取一帧
        ret, frame = vs.read()
        img = cv2.resize(frame, size)
        pipe.stdin.write(img.tobytes())
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    # 关闭窗口
    cv2.destroyAllWindows()

    # 停止读取
    vs.release()
