import glob
import os
import pickle
import threading
import webbrowser
from os import walk
from timeit import default_timer as timer

import cv2
import numpy as np
import werkzeug
from flask import (Flask, flash, json, jsonify, redirect, render_template,
                   request, url_for)
from PIL import Image
from werkzeug.utils import secure_filename

# import yolo
import bleu_rating
import predict_approach1
#import predict_approach2

app = Flask(__name__)
mypath = 'instance/test/'

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
    app.run(host= '0.0.0.0', port=5000, debug=False)    
