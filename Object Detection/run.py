from flask import Flask,flash,jsonify,request,render_template,redirect,url_for
from PIL import Image
import numpy
import cv2
import time
import werkzeug
import os

app = Flask(__name__)

objs = []

@app.route("/")
def firstpage():
    global objs
    return render_template('index.html',objs=objs)

@app.route("/getobjects",methods=["GET", "POST"])
def objectdetection():
    #import maskrcnn
    import yolo
    global objs
    file = request.files['file']
    inputimg = Image.open(file).convert('RGB')
    img = numpy.array(inputimg)

    cv2.imwrite('static/flaskimgs/flaskimg.jpg',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
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

    start = time.time()
    yolo_objs = yolo.getobj(img)
    end = time.time()
    print('YOLO V3:')
    print(yolo_objs)
    print('Time:',end-start)
    print()
    yolotime = (end-start)
    # ykeys, yvalues = zip(*yolo_objs.items())
    

    # maskout = cv2.imread('static/images/maskrcnn_out.png')
    # yoloout = cv2.imread('static/images/yolo_out.png')
    # return render_template('display.html',maskout=maskout, yoloout=yoloout)

    return render_template('display.html',yolotime=round(yolotime,3),yolo_objs=yolo_objs)

    #return render_template('display.html',yolotime=round(yolotime,3),masktime=round(masktime,3),maskrcnn_objs=maskrcnn_objs,yolo_objs=yolo_objs)

if __name__ == "__main__":
    app.run(host= '0.0.0.0', port=5000, debug=True)
