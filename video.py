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
### blpr pipeline for images
# ---------------------------------
def blpr_video(source_path, save_path="{HOME}/inferences", show_detection=False):
    """
    source_path: pass the full path along with the file extension
    save_path: folder to which the detection is to be saved, by default saved to "./Project_BLPR/inferences/" with name = "{source_file_name} - detections.avi"
    show_detection: by default False. If True then will show the detection frame by frame as it happens. Caution - might be very slow

    Returns: saved file location
    """
    capture = cv.VideoCapture()
    if capture.isOpened() is not True:
        capture.open()

    cv.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cv.set(cv.CAP_PROP_FRAME_HEIGHT, 640)

    source_file_name = os.path.splitext(os.path.basename(f"{source_path}"))[0]
    save_detections_to = f"{save_path}/{source_file_name} - detections.avi"

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter(save_detections_to, fourcc, 20.0, (640, 640))

    while True:
        ret, frame = capture.read()

        if ret is False and save_detections_to:
            return f"Detections saved to {save_detections_to}. Exiting..."

        if ret is False:
            return "Could not read any frame. Stream may have ended. Exiting..."

        blur_frame = cv.GaussianBlur(frame, (5, 5), 0)
        car_text = blp_text_extraction_pipeline(blur_frame)

        if car_text is None or car_text.strip() == "":
            car_text = "Sorry, Couldn't detect any number plate text."

        ### save detection
        # drawing 'text' over the original frame
        cv.putText(
            frame,
            car_text,
            (20, 110),
            cv.FONT_ITALIC,
            fontScale=3,
            color=(0, 0, 255),
            thickness=3,
        )

        # writing to detection.avi
        out.write(frame)

        ### show detections as it happens
        if show_detection:
            cv.imshow(f"Detections for {source_path} as it happens", frame)

            if cv.waitKey(1) == ord("q"):
                show_detection = False
                cv.destroyAllWindows()

        capture.release()
        out.release()
        cv.destroyAllWindows()
