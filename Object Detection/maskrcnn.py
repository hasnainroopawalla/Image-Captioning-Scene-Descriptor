from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2 as cv2
import os
import skimage.io
from keras import backend as K

def getobj(image):
  K.clear_session()
  class_names = ['BG', 'person', 'bicycle', 'car', 'motorbike', 'aeroplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
                'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

  def id_to_class(obj_ids):
    objs = {}
    obj_ids = dict((x,list(obj_ids).count(x)) for x in set(obj_ids))
    for i in obj_ids:
      objs[class_names[i]] = obj_ids[i]
    return objs
      
  ROOT_DIR = os.path.abspath("")
  MODEL_DIR = os.path.join(ROOT_DIR, "logs")
  COCO_MODEL_PATH = "../weights/mask_rcnn_coco.h5"


  class SimpleConfig(Config):

    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(class_names)

  config = SimpleConfig()

  model = modellib.MaskRCNN(mode="inference", config=config,
    model_dir=os.getcwd())
  print(model.summary())
  model.load_weights(COCO_MODEL_PATH, by_name=True)


  #image = skimage.io.imread('plane.jpg')

  results = model.detect([image], verbose=1)
  objs = id_to_class(results[0]['class_ids'])
  K.clear_session()
  
  r = results[0]
  a = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                              class_names, r['scores'])
  a.savefig('static/output/maskrcnn_out.png', bbox_inches='tight')
  return objs
