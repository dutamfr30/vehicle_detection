from ultralytics import YOLO
import cv2
import time
import numpy as np 

mode = 'predict'
model = YOLO('yolov8n.pt')
conf = 0.50
classes = 2,3,5,7

cam = cv2.VideoCapture('D:\TUGAS AKHIR DUTA\Dataset kendaraan (2)\challenge_video.mp4')


while True:
    ret, image = cam.read()
    if not ret:
        break


    waktu_mulai = time.time()
    frame = cv2.resize(image, (640, 480))
    result = model.predict(frame, show=True)
   

    print("waktu", time.time()-waktu_mulai)
 

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
