from flask import Flask,flash,jsonify,request,render_template,redirect,url_for
from PIL import Image
import numpy
import cv2

app = Flask(__name__)

objs = []

@app.route("/")
def firstpage():
    global objs
    return render_template('index.html',objs=objs)

@app.route("/getobjects",methods=["GET", "POST"])
def newimage():
    import detect
    global objs
    file = request.files['file']
    img = Image.open(file).convert('RGB')
    image1 = numpy.array(img) 
    objs = detect.getobj(image1)
    print(objs)
    # b = Image.open('static/images/out.png')
    # # b.show()
    return redirect('/')

if __name__ == "__main__":
    #threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:9000")).start()
    app.run(host= '0.0.0.0', port=5000, debug=True)
