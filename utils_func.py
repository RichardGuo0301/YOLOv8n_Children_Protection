import glob
import os

import cv2


def remove_images():
    files = glob.glob('images/*.jpg')
    for i in files:
        os.remove(i)


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def resize_image(image, percent=0.5):
    h, w, _ = image.shape
    h, w = int(h * percent), int(w * percent)
    image = cv2.resize(image, (w, h))
    return image, h, w


def resize_original(image, height, width):
    image = cv2.resize(image, (width, height))
    return image
