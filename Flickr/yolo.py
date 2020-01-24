import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import string 
import random
import glob

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    objs = []
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            
            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            objs.append(text.split(':')[0])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    objs = dict((x,list(objs).count(x)) for x in set(objs))
    #print(objs)
    return img, objs


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels, confidence, threshold, show_time, 
            boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                        swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        if show_time:
            print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        
    # Draw labels and boxes on the image
    objs = []
    img, objs = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs, objs



FLAGS = []
WEIGHTS_PATH = '../../weights/yolov3-coco/yolov3.weights'
CONFIG_PATH = '../../weights/yolov3-coco/yolov3.cfg'
LABELS_PATH = '../../weights/yolov3-coco/coco-labels'
confidence = 0.5
threshold = 0.3
show_time = False

# Get the labels
labels = open(LABELS_PATH).read().strip().split('\n')

# Intializing colors to represent each label uniquely
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Load the weights and configutation to form the pretrained YOLOv3 model
net = cv.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

# Get the output layer names of the model
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def getobj(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    img, _, _, _, _, objs = infer_image(net, layer_names, height, width, img, colors, labels, confidence, threshold, show_time)

    files = glob.glob('static/images/yolo_out/*') # Clear yolo_out folder contents
    for f in files:
        os.remove(f)

    res = ''.join(random.choices(string.ascii_uppercase +  #Generate random file name
                             string.digits, k = 7)) 
    output_path = 'static/images/yolo_out/'+str(res)+'.jpg'
    cv.imwrite(output_path, cv.cvtColor(img,cv.COLOR_BGR2RGB))

    return objs, output_path

