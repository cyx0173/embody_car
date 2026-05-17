import cv2
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

if not cap0.isOpened():
    raise RuntimeError("无法打开摄像头 0")

if not cap1.isOpened():
    raise RuntimeError("无法打开摄像头 1")

rotate_code_cap1 = cv2.ROTATE_90_COUNTERCLOCKWISE

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0:
        print("摄像头 0 读取失败")
        break

    if not ret1:
        print("摄像头 1 读取失败")
        break

    # cap0：不旋转，直接检测
    results0 = model(frame0, verbose=False)
    annotated0 = results0[0].plot()

    # cap1：旋转 90 度后检测
    frame1_rotated = cv2.rotate(frame1, rotate_code_cap1)
    results1 = model(frame1_rotated, verbose=False)
    annotated1 = results1[0].plot()

    cv2.imshow("Camera 0 YOLO Detection", annotated0)
    cv2.imshow("Camera 1 YOLO Detection Rotated 90", annotated1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()