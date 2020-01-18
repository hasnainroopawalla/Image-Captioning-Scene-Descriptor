from flask import Flask,flash,jsonify,request,render_template,redirect,url_for
from PIL import Image
import numpy
import cv2
import time
import werkzeug
import os
import predict

app = Flask(__name__)

objs = []

@app.route("/")
def firstpage():
    return render_template('index.html')

@app.route("/process_img",methods=["GET", "POST"])
def objectdetection():
    file = request.files.getlist('files[]')[0]
    inputimg = Image.open(file).convert('RGB')
    img = numpy.array(inputimg)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite('input_img.png',img)

    return jsonify({'caption' : 'Files successfully uploaded'})

if __name__ == "__main__":
    app.run(host= '0.0.0.0', port=5000, debug=True)

    # import maskrcnn
    # import yolo
    # global objs
    # file = request.files['file']
    # inputimg = Image.open(file).convert('RGB')
    # img = numpy.array(inputimg)

    # cv2.imwrite('static/flaskimgs/flaskimg.jpg',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
    # print()
    # start = time.time()
    # maskrcnn_objs = maskrcnn.getobj(img)
    # end = time.time()
    # print('\nMask-RCNN:')
    # print(maskrcnn_objs)
    # print('Time:',end-start)
    # print()
    # masktime = (end-start)
    # # mkeys, mvalues = zip(*maskrcnn_objs.items())

    # start = time.time()
    # yolo_objs = yolo.getobj(img)
    # end = time.time()
    # print('YOLO V3:')
    # print(yolo_objs)
    # print('Time:',end-start)
    # print()
    # yolotime = (end-start)
    # # ykeys, yvalues = zip(*yolo_objs.items())
    

    # # maskout = cv2.imread('static/images/maskrcnn_out.png')
    # # yoloout = cv2.imread('static/images/yolo_out.png')
    # # return render_template('display.html',maskout=maskout, yoloout=yoloout)

