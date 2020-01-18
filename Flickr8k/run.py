import os
import pickle

import cv2
import numpy as np
import werkzeug
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   url_for)
from PIL import Image

import predict

app = Flask(__name__)

input_img_path = 'static/images/input_img.jpg'

@app.route("/")
def firstpage():
    return render_template('index.html')

@app.route("/process_img",methods=["GET", "POST"])
def objectdetection():
    file = request.files.getlist('files[]')[0]
    inputimg = Image.open(file).convert('RGB')
    img = np.array(inputimg)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(input_img_path,img)
    caption = predict.predict_caption(input_img_path, is_test=0)
    return jsonify({'caption' : caption})

if __name__ == "__main__":
    app.run(host= '0.0.0.0', port=5000, debug=False)
