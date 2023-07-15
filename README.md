# Project_BLPR 
In partial fulfillment of the requirements for the degree of Bachelor of Science in Mechanical Engineering.

---------------------------------
## Title of the Project
### Development of A Multi-Stage Deep Learning Pipeline for Automatic Bangla License Plate Recognition.

-----------------------------------
## Abstract 

ALPR (Automatic License Plate Recognition) is an important application of computer vision. Despite being the eighth largest populous country in the world, there has been little progress in developing real-time BLPR (Bangla License Plate Recognition) systems. Most of the existing research uses two common steps, license plate bounding box recognition and EasyOCR for Bangla character recognition. The accuracy of EasyOCR usually varies between 75-78% at most [1] and one paper even reported an accuracy as low as 60% [2]. Another problem is that its accuracy can’t be readily increased. So, in this research project, we trained two custom YOLO-V8 models. The first was used for detecting the license plate bounding box and the second was for extracting text from the license plates. The first model achieved an Accuracy of 97.8% (Precision - 0.983, Recall - 0.99, F1 Score - 0.986) on the test dataset. The second model has 100 classes and it achieved an Average Accuracy of about 93% or more across the classes. With a CPU, end-to-end text extraction from images took about 0.3 - 0.55 sec, and with a GPU, end-to-end text extraction from images took 30-80 ms, depending on the image size. These are some considerable improvements over the previous works, such as [3]. The author hopes that this end-to-end deep learning framework can open new doors to BLPR systems and their implementation in real-life scenarios.

-------------------------------
## Note

**This repo contains the notebooks used to train the custom yolo-v8 models (along with the download links for the datasets used). Here we implemented and tested the developed BLPR system for image inputs (image.py) and pre-recorded video inputs (video.py) and made sure that the pipelines works without any issues. But we couldn't verify the pipeline developed for live video inputs (live_capture.py) as no good quality external video-capturing device was available at the time. But it should work seamlessly since it is essentially the same as the video.py pipeline.**

--------------------------------------

## General Outline of the End-to-End BLPR pipeline

![outline_of_methodology](https://github.com/MaidulHasan/Project_BLPR/assets/71931144/936b928e-cbab-4714-a33d-97cc861eb652)

### Outline of the License Plate Bounding Box Detector Model**

![outline_of_model_lp_bb_detector](https://github.com/MaidulHasan/Project_BLPR/assets/71931144/569a02ec-6094-4764-8d1e-b21e9e54eb43)

### Outline of Bangla License Plate Text Extractor Model**

![outline_of_model_blp_text_extractor](https://github.com/MaidulHasan/Project_BLPR/assets/71931144/f8dfccb1-de26-43ce-9947-cd615f836e59)

---------------------------------------
## References

[1] M. Tusar, M. Bhuiya, M. Hossain, A. Tabassum, and R. Khan, Real Time Bangla License Plate Recognition with Deep Learning Techniques. 2022, p. 6. doi: 10.1109/IICAIET55139.2022.9936764.

[2] A. Ashrafee, A. M. Khan, M. S. Irbaz, and M. A. A. Nasim, “Real-time Bangla License Plate Recognition System for Low Resource Video-based Applications.” arXiv, Nov. 14, 2021. doi: 10.48550/arXiv.2108.08339.

[3] M. S. H. Onim et al., “BLPnet: A new DNN model and Bengali OCR engine for Automatic License Plate Recognition.” arXiv, Feb. 18, 2022. Accessed: May 18, 2023. [Online]. Available: http://arxiv.org/abs/2202.12250
