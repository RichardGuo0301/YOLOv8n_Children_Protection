from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils_func import make_dir

path = 'models/door_model4.pt'
door_model = YOLO(path)
door_model.conf = 0.45
people_model = YOLO('models/yolov8n.pt')


def detect_door(img_path, frame):
    results = door_model(img_path)
    door = []
    annotator = Annotator(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            # print(b)
            conf = r.boxes.conf.tolist()
            door.append(b)
            c = box.cls
            annotator.box_label(b, door_model.names[int(c)], color=(255, 128, 128))
    frame = annotator.result()
    return door, frame


def detect_people(img_path, frame):
    imgs = [img_path]  # batch of images
    people = []
    # Inference
    results = people_model(imgs, classes=0)
    annotator = Annotator(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            # print(b)
            conf = r.boxes.conf.tolist()
            people.append(b)
            c = box.cls
            annotator.box_label(b, people_model.names[int(c)], color=(128, 128, 255))
    frame = annotator.result()
    return people, frame


def write_next2door(image):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method

    image = cv2.putText(image, 'Person -', (5, 50), font, fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, 'Next to door', (5, 80), font, fontScale, color, thickness, cv2.LINE_AA)
    return image


def next2door(person_box, door_box, frame):
    epsilon = 5
    if door_box[0] - epsilon <= person_box[0] <= door_box[2] + epsilon and door_box[1] - epsilon <= \
            person_box[
                1] <= door_box[3] + epsilon:
        return write_next2door(frame), True

    if door_box[0] - epsilon <= person_box[2] <= door_box[2] + epsilon and door_box[1] - epsilon <= \
            person_box[1] <= door_box[3] + epsilon:
        return write_next2door(frame)

    if door_box[0] - epsilon <= person_box[0] <= door_box[2] + epsilon and door_box[1] - epsilon <= \
            person_box[1] <= door_box[3] + epsilon:
        return write_next2door(frame), True

    if door_box[0] - epsilon <= person_box[2] <= door_box[2] + epsilon and door_box[1] - epsilon <= \
            person_box[3] <= door_box[1] + epsilon:
        return write_next2door(frame), True
    return frame, False


def process_image(filename, frame, door=[], save_image=False):
    temp_door, frame = detect_door(filename, frame)
    people_list, frame = detect_people(filename, frame)
    window_list = []  # This is the code that you use to detect the window
    is_near_door = False
    if temp_door:
        door = temp_door
    if door and people_list:
        for p in people_list:
            frame, is_near_door = next2door(person_box=p.tolist(), door_box=door[0].tolist(), frame=frame)
    if save_image:
        make_dir('result')
        cv2.imwrite(f'result/{Path(filename).stem}.jpg', frame)
    # cv2.imshow('image', frame)
    return frame, len(people_list), len(temp_door), len(window_list), is_near_door


if __name__ == '__main__':
    path = 'sample_images/door6.jpg'
    img = cv2.imread(path)
    process_image(filename=path, frame=img, save_image=True)
