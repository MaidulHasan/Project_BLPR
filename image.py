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
def blpr_image(source_path, save_detection=True, save_path=f"{HOME}/inferences"):
    """
    source_path: pass the full path along with the image extension
    save_detection: by default True
    save_path: folder to which the detection is to be saved, by default saved to "./Project_BLPR/inferences/" with filename = "{source_file_name} - {car_text}.png"

    Returns: the extracted text from the car number plate
    """

    source_img = cv.imread(source_path)
    frame = source_img.copy()
    blur_img = cv.GaussianBlur(source_img, (5, 5), 0)
    car_text = blp_text_extraction_pipeline(blur_img)

    if car_text is None or car_text.strip() == "":
        car_text = "Sorry, Couldn't detect any number plate text."
        print(car_text)
        return car_text

    ### save detection
    if save_detection:
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

        # saving image
        source_file_name = os.path.splitext(os.path.basename(f"{source_path}"))[0]
        cv.imwrite(f"{save_path}/{source_file_name} - {car_text}.png", frame)

    ### return the extracted number plate
    return car_text
