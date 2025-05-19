!pip install ultralytics opencv-python
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
from google.colab import files
uploaded = files.upload()
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("parking1.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

   
    resized = cv2.resize(annotated_frame, (640, 360))
    cv2_imshow(resized)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
