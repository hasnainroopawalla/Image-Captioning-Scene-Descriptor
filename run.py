from flask import Flask,flash,jsonify,request,render_template,redirect,url_for
from PIL import Image
import numpy

app = Flask(__name__)

@app.route("/")
def firstpage():
    return render_template('index.html')

@app.route("/getobjects",methods=["GET", "POST"])
def newimage():
    import detect
    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img.show() 
    image1 = numpy.array(img) 
    print(detect.getobj(image1))
    return redirect('/')

if __name__ == "__main__":
    #threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:9000")).start()
    app.run(host= '0.0.0.0', port=5000, debug=True)
