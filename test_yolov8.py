import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

imageName = 'sample_images/door4.jpg'
frame = cv2.imread(imageName)
# Load a model
model = YOLO("models/door_model4.pt")
results = model(imageName)
print(results)
for r in results:

    annotator = Annotator(frame)

    boxes = r.boxes
    print(r.boxes.conf.tolist())
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        print(b)
        c = box.cls
        annotator.box_label(b, model.names[int(c)],color=(255, 255, 128))

frame = annotator.result()
cv2.imshow('YOLO V8 Detection', frame)
cv2.waitKey(0)
