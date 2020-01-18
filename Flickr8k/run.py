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

""" from flask import Flask

UPLOAD_FOLDER = ''

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
import os
#import magic
import urllib.request

from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('tt.html')

@app.route('/python-flask-files-upload', methods=['POST'])
def upload_file():
	# check if the post request has the file part
	if 'files[]' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	
	files = request.files.getlist('files[]')
	
	errors = {}
	success = False

	for file in files:

		if file and allowed_file(file.filename):
            
			filename = secure_filename(file.filename)
            
			#file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #inputimg = Image.open(file).convert('RGB')
            img = numpy.array(inputimg)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imshow('input_img',img)
            cv2.waitKey(0)
			success = True
		else:
			errors[file.filename] = 'File type is not allowed'
	
	if success and errors:
		errors['message'] = 'File(s) successfully uploaded'
		resp = jsonify(errors)
		resp.status_code = 206
		return resp
	if success:
		resp = jsonify({'message' : 'Files successfully uploaded'})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify(errors)
		resp.status_code = 400
		return resp

if __name__ == "__main__":
    app.run() """