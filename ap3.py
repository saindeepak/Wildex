import sys,os,glob,re
import numpy as np
import json
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from flask import Flask,request,redirect,url_for,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer     

app = Flask(__name__, static_url_path = "", static_folder = "templates")

model_path = 'keras.h5'

model = load_model(model_path)
model._make_predict_function()


def model_predict(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x,mode='caffe')
    preds = model.predict(x)
    return preds



@app.route('/',methods=['GET'])
def index():
    return render_template('index1.html')

@app.route('/analyze', methods=["POST"])
def analyze():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname('uploads')
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        
        preds = model_predict(file_path, model)
        pred_class = decode_predictions(preds,top=1)
        result = str(pred_class[0][0][1])
        #return result

        return json.dumps({"image": result})
    return None

if __name__ == '__main__':
    app.run()