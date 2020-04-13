#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from keras.layers import Input
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
yolo = YOLO()
# 调用摄像头
capture=cv2.VideoCapture(0) # capture=cv2.VideoCapture("1.mp4")

while(True):
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(yolo.detect_image(frame))

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    cv2.imshow("video",frame)
    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        break

yolo.close_session()
    