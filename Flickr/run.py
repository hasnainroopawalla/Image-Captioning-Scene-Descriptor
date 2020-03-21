import os
import pickle
import threading
import webbrowser
from timeit import default_timer as timer
import glob
import cv2
import numpy as np
import werkzeug
from flask import (Flask, flash, json, jsonify, redirect, render_template, request,
                   url_for)
from PIL import Image
import werkzeug
from os import walk
from werkzeug.utils import secure_filename

# import yolo
import bleu_rating
import predict_approach1
import predict_approach2

app = Flask(__name__)

input_img_path = 'static/images/input_img.jpg'
mypath = 'instance/test/'

@app.route("/")
def firstpage():
    return render_template('index.html')

@app.route("/process_img",methods=["GET", "POST"])
def objectdetection():

    bleu_1 = ''
    bleu_2 = ''

    preprocess_flag = request.form['preprocess']
    searchtype = request.form['searchtype']
    file_name = request.form['file_name']

    file = request.files.getlist('files[]')[0]
    inputimg = Image.open(file).convert('RGB')
    img = np.array(inputimg)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(input_img_path,img)

    # start_yolo = timer()
    # yolo_objs, yolo_img = yolo.getobj(img)
    # end_yolo = timer()
    if 'Yes' in preprocess_flag:
        preprocess_flag = 0
    else:
        preprocess_flag = 1

    
    start_1 = timer()
    caption_1, acc_1 = predict_approach1.predict_caption(input_img_path, preprocess_flag, searchtype)
    end_1 = timer()

    start_2 = timer()
    caption_2, acc_2 = predict_approach2.predict_caption(input_img_path, preprocess_flag, searchtype)
    end_2 = timer()

    if preprocess_flag == 1:
        bleu_1 = bleu_rating.get_bleu(file_name,caption_1)
        bleu_2 = bleu_rating.get_bleu(file_name,caption_2)

    str_acc_1 = []
    str1 = " "  
    for i in acc_1:
        str_acc_1.append(str(round(i*100,2)))

    str_acc_2 = []
    str2 = " "  
    for i in acc_2:
        str_acc_2.append(str(round(i*100,2)))

    return jsonify({'caption_1':caption_1, 'acc_1':str1.join(str_acc_1),'caption_2':caption_2, 'acc_2':str2.join(str_acc_2), 'time_1':round(end_1-start_1,2), 'time_2':round(end_2-start_2,2), 'bleu_1':bleu_1, 'bleu_2':bleu_2})#'yolo_time':round(end_yolo-start_yolo,2), 'yolo_img':yolo_img})


@app.route("/a",methods=["GET", "POST"])
def hello_world():
       
    files = glob.glob(mypath+'*')
    for f in files:
        os.remove(f)

    os.makedirs(os.path.join(app.instance_path, 'test'), exist_ok=True)
    if request.method =='POST':
        file = request.files['pic']
        if file:
            filename = file.filename
            file.save(os.path.join(app.instance_path, 'test', secure_filename(file.filename)))

            f = []
            for (dirpath, dirnames, filenames) in walk(mypath):
                f.extend(filenames)
            input_img_path = mypath+f[0]

            start_1 = timer()
            caption_1, acc_1 = predict_approach1.predict_caption(input_img_path, preprocess_flag=0, searchtype='Greedy')
            end_1 = timer()

        time_taken = str(round(end_1 - start_1,2))
        caption = caption_1+' ( '+time_taken+' seconds )'
        print(caption)
    return json.dumps({'caption': caption})  


if __name__ == "__main__":
    threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host= '0.0.0.0', port=5000, debug=False)
