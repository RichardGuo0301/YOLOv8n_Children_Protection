import time
from datetime import datetime

import cv2
import pytz

from database import Database
from image import process_image
from utils_func import remove_images

db = Database()


def process_video(video_path=0):
    pool = []
    device = 'Device 1'
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    # Open the default camera (camera with index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    # Define the window name
    window_name = "Video Stream"
    i = 0
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        if ret:
            filename = f'images.jpg'
            cv2.imwrite(filename, frame)

            frame, n_people, n_doors, n_windows, is_near_door = process_image(filename, frame, pool)
            cv2.imwrite(filename, frame)
            cv2.imshow('frame', frame)
            # Set the time zone to Pacific Time
            pacific_time_zone = pytz.timezone('US/Pacific')
            # Get the current time in the Pacific Time Zone
            current_time_pacific = str(datetime.now(pacific_time_zone))[:19]
            print(current_time_pacific)
            # print("Current time in Pacific Time:", current_time_pacific)
            url = db.upload_file(firebase_path=f"folder_name/img{i%10}.jpg", local_path=filename)
            # Edit the near_window value with your value when detected window
            data = {'Time': current_time_pacific, 'img_url': url, 'near_door': is_near_door,
                    'near_window': False, 'people_detected': n_people, 'door_detected': n_doors,
                    'window_detected': n_windows}
            print(data)
            db.set_events(data=data, device=device, collection='events')
            time.sleep(2)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            i += 1
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    remove_images()


if __name__ == '__main__':
    process_video()
