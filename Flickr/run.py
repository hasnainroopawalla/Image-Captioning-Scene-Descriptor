import os
import pickle

import cv2
import numpy as np
import werkzeug
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   url_for)
import threading, webbrowser
from PIL import Image

import predict_approach1
import predict_approach2
from timeit import default_timer as timer
import yolo

app = Flask(__name__)

input_img_path = 'static/images/input_img.jpg'

@app.route("/")
def firstpage():
    return render_template('index.html')

@app.route("/process_img",methods=["GET", "POST"])
def objectdetection():

    preprocess_flag = request.form['preprocess']
    searchtype = request.form['searchtype']

    file = request.files.getlist('files[]')[0]
    inputimg = Image.open(file).convert('RGB')
    img = np.array(inputimg)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(input_img_path,img)

    start_yolo = timer()
    yolo_objs, yolo_img = yolo.getobj(img)
    end_yolo = timer()
    
    start_1 = timer()
    caption_1, acc_1 = predict_approach1.predict_caption(input_img_path, preprocess_flag, searchtype)
    end_1 = timer()

    start_2 = timer()
    caption_2, acc_2 = predict_approach2.predict_caption(input_img_path, preprocess_flag, searchtype)
    end_2 = timer()

    str_acc_1 = []
    str1 = " "  
    for i in acc_1:
        str_acc_1.append(str(round(i*100,2)))

    str_acc_2 = []
    str2 = " "  
    for i in acc_2:
        str_acc_2.append(str(round(i*100,2)))

    return jsonify({'caption_1':caption_1, 'acc_1':str1.join(str_acc_1),'caption_2':caption_2, 'acc_2':str2.join(str_acc_2), 'time_1':round(end_1-start_1,2), 'time_2':round(end_2-start_2,2)})#'yolo_time':round(end_yolo-start_yolo,2), 'yolo_img':yolo_img})

if __name__ == "__main__":
    threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host= '0.0.0.0', port=5000, debug=False)
