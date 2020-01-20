import os
import pickle

import cv2
import numpy as np
import werkzeug
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   url_for)
import threading, webbrowser
from PIL import Image

import predict
from timeit import default_timer as timer

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
    start = timer()

    caption, acc = predict.predict_caption(input_img_path, preprocess_flag, searchtype)

    end = timer()
    str_acc = []
    str1 = " "  
    for i in acc:
        str_acc.append(str(round(i*100,2)))

    return jsonify({'caption':caption, 'acc':str1.join(str_acc), 'time':round(end-start,2)})

if __name__ == "__main__":
    threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host= '0.0.0.0', port=5000, debug=False)
