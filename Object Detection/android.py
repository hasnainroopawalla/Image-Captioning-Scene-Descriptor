import flask
import time
import cv2
import os

app = flask.Flask(__name__)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    files_ids = list(flask.request.files)
    print("\nNumber of Received Images : ", len(files_ids))
    image_num = 1
    for file_id in files_ids:
        print("\nSaving Image ", str(image_num), "/", len(files_ids))
        imagefile = flask.request.files[file_id]
    
        
        imagefile.save('static/androidimgs/androidimg.jpg')
        img = load_images_from_folder('static/androidimgs')[0]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        import maskrcnn
        import yolo
        global objs

        print()
        start = time.time()
        maskrcnn_objs = maskrcnn.getobj(img)
        end = time.time()
        print('\nMask-RCNN:')
        print(maskrcnn_objs)
        print('Time:',end-start)
        print()
        
        start = time.time()
        yolo_objs = yolo.getobj(img)
        end = time.time()
        print('YOLO V3:')
        print(yolo_objs)
        print('Time:',end-start)
        print()
    ##    maskrcnn_out = cv2.imread('static/images/maskrcnn_out.png')
    ##    yolo_out = cv2.imread('static/images/yolo_out.png')


        
    return "Image(s) Uploaded Successfully. Come Back Soon."

app.run(host="0.0.0.0", port=5000, debug=True)
