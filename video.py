# ---------------------------------
### Imports
# ---------------------------------

import numpy as np
import cv2 as cv
import os
from blpr_pipeline import blp_text_extraction_pipeline


# ---------------------------------
### define HOME
# ---------------------------------

HOME = "/content/drive/MyDrive/Project_BLPR"
os.chdir(HOME)


# ---------------------------------
### blpr pipeline for pre-recorded video inputs
# ---------------------------------
def blpr_video(source_path, save_path=f"{HOME}/inferences"):
    """
    source_path: pass the full path along with the file extension
    save_path: folder to which the detection is to be saved, by default saved to "./Project_BLPR/inferences/" with name = "blpr_video_detections-{source_file_name}.mp4"

    Returns: saved file location
    """
    capture = cv.VideoCapture(source_path)
    if capture.isOpened() is not True:
        capture.open()

    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    source_file_name = os.path.splitext(os.path.basename(f"{source_path}"))[0]
    save_detection_to = f"{save_path}/blpr_video_detections-{source_file_name}.mp4"

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(save_detection_to, fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = capture.read()

        if ret is False:
            print("Could not read any frame. Stream may have ended. Exiting...")
            break

        blur_frame = cv.GaussianBlur(frame, (5, 5), 0)
        car_text = blp_text_extraction_pipeline(blur_frame)

        ### save detection
        # drawing 'text' over the original frame
        if car_text is not None and car_text.strip() != "":
            cv.putText(
                frame,
                car_text,
                (20, 110),
                cv.FONT_ITALIC,
                fontScale=2,
                color=(0, 0, 255),
                thickness=2,
            )

        # writing to detection.mp4
        out.write(frame)

    capture.release()
    out.release()
    cv.destroyAllWindows()

    return save_detection_to
