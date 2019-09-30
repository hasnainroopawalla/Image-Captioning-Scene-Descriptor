from flask import Flask,flash,jsonify,request,render_template,redirect,url_for
from PIL import Image
import numpy
import cv2
import time

app = Flask(__name__)

objs = []

@app.route("/")
def firstpage():
    global objs
    return render_template('index.html',objs=objs)

@app.route("/getobjects",methods=["GET", "POST"])
def newimage():
    import maskrcnn
    import yolo
    global objs
    file = request.files['file']
    inputimg = Image.open(file).convert('RGB')
    img = numpy.array(inputimg)
    
    start = time.time()
    maskrcnn_objs = maskrcnn.getobj(img)
    end = time.time()
    print('Mask-RCNN:')
    print(maskrcnn_objs)
    print('Time:',end-start)
    
    print()
    
    start = time.time()
    yolo_objs = yolo.getobj(img)
    end = time.time()
    print('YOLO V3:')
    print(yolo_objs)
    print('Time:',end-start)
    
    # b = Image.open('static/images/out.png')
    # # b.show()
    return redirect('/')

if __name__ == "__main__":
    #threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:9000")).start()
    app.run(host= '0.0.0.0', port=5000, debug=True)
