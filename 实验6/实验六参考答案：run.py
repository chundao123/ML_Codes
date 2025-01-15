import argparse
import cv2
from kcf import Tracker

if __name__ == '__main__':
    #打开视频文件
    cap = cv2.VideoCapture('car.avi')
    # cap = cv2.VideoCapture('tokyo-walk.mp4')
    
    #获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    #设置播放速度
    new_fps = 0.5*fps
    cap.set(cv2.CAP_PROP_FPS,new_fps)
    ok, frame = cap.read()
    if not ok:
        print("error reading video")
        exit(-1)
    roi = cv2.selectROI("tracking", frame, False, False)

    tracker = Tracker()
    tracker.init(frame, roi)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        x, y, w, h = tracker.update(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c==27 or c==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()