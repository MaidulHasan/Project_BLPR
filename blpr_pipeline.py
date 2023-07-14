# ---------------------------------
### Imports
# ---------------------------------

import numpy as np
import os

from ultralytics import YOLO

"""
if ultralytics is not installed use,
Pip install method (recommended)
%cd
!pip install ultralytics==8.0.112
!pip install dill
display.clear_output()
"""


# ---------------------------------
### define HOME
# ---------------------------------

HOME = "/content/drive/MyDrive/Project_BLPR"

inference_folder = f"{HOME}/inferences"
try:
    os.mkdir(inference_folder)
except FileExistsError:
    pass
os.chdir(HOME)

# ------------------------------------
### custom models instantiation
# ------------------------------------

path_to_model_lp_bb_detection_best_pt = (
    f"{HOME}/model_lp_bb_detection/lp_bb_trained_weights/train/weights/best.pt"
)

model_lp_detection = YOLO(path_to_model_lp_bb_detection_best_pt)

path_to_model_blp_text_extraction_best_pt = (
    f"{HOME}/model_blp_text_extraction/blp_text_trained_weights/train/weights/best.pt"
)

model_blp_text_extraction = YOLO(path_to_model_blp_text_extraction_best_pt)


# --------------------------------------------------------------------
### Extract license plate text from a particular frame or image
# ---------------------------------------------------------------------
def blp_text_extraction_pipeline(frame):
    ### license plate bounding box

    results_lp_bb = model_lp_detection.predict(source=frame, conf=0.4)[0]
    orig_img = results_lp_bb.orig_img

    xyxy_boxes = results_lp_bb.boxes.xyxy.cpu().numpy().astype(np.int16)
    if xyxy_boxes.any() == False:
        return None

    # for cropping out the license plate bounding box
    # ndarray[y_start:y_end, x_start:x_end]
    lp_bb_crop = orig_img[
        xyxy_boxes[:, 1][0] : xyxy_boxes[:, 3][0],
        xyxy_boxes[:, 0][0] : xyxy_boxes[:, 2][0],
    ]

    ### text extraction

    results_blp_text = model_blp_text_extraction.predict(
        lp_bb_crop, show_conf=False, agnostic_nms=True
    )[0]

    # LP Text classes
    class_names = results_blp_text.names
    class_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    class_vehicle_type = [
        10,
        11,
        17,
        20,
        25,
        29,
        30,
        33,
        36,
        39,
        41,
        43,
        46,
        50,
        51,
        57,
        60,
        68,
        77,
        87,
        89,
        95,
        96,
        98,
        100,
    ]
    class_metro = [65]

    # id of detected classes, width of detected bounding boxes, x center of detected bounding boxes
    detected_classes = [
        cls_id for cls_id in results_blp_text.boxes.cls.cpu().numpy().astype(np.int16)
    ]
    detected_classes_box_width = (
        results_blp_text.boxes.xywh.cpu().numpy()[:, 2].astype(np.int16).tolist()
    )
    detected_classes_box_x_center = (
        results_blp_text.boxes.xywh.cpu().numpy()[:, 0].astype(np.int16).tolist()
    )

    # pos_metro: string defining whether 'metro' or not
    if 65 in detected_classes:
        pos_metro = class_names.get(65)
        id_65_pos_in_detected_classes_lst = detected_classes.index(65)
        detected_classes.pop(id_65_pos_in_detected_classes_lst)
        detected_classes_box_x_center.pop(id_65_pos_in_detected_classes_lst)
        detected_classes_box_width.pop(id_65_pos_in_detected_classes_lst)
    else:
        pos_metro = ""

    # pos_loc
    detected_id_w_x = list(
        zip(detected_classes, detected_classes_box_width, detected_classes_box_x_center)
    )
    id_w_x_sorted_by_w = sorted(detected_id_w_x, key=lambda x: -x[1])

    try:
        pos_loc = class_names.get(id_w_x_sorted_by_w[0][0])
        id_w_x_sorted_by_w.pop(0)
    except:
        pos_loc = ""

    # pos_number: string of detected numbers in order & pos_type: string of detected vehicle type
    id_w_x_sorted_by_x = sorted(id_w_x_sorted_by_w, key=lambda x: x[2])

    pos_number = ""
    pos_type = ""

    for id, _, _ in id_w_x_sorted_by_x:
        if id in class_number:
            pos_number += class_names.get(id)
        if id in class_vehicle_type:
            pos_type = class_names.get(id)

    # car_text: extracted text from the number plate
    car_text = f"{pos_loc} {pos_metro} {pos_type} {pos_number}"

    return car_text
