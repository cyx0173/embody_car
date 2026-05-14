import cv2
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 在这里旋转每一帧图像
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("YOLO11 Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()