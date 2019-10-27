# Image-Caption
An image captioning model which uses a Mask R-CNN to detect objects and an LSTM to recursively generate a caption. This gramatically correct sentence can accurately describe the scene of an image, enabling any individual to visualize the image mentally

In CMD run 'run.py'
Navigate to 'localhost:5000' from a browser

Currently we have only deployed the object detection module using Mask-RCNN and YOLO V3 (for comparison).

The parent folder of this repository should contain the trained Mask-RCNN and YOLO V3 model weights.
