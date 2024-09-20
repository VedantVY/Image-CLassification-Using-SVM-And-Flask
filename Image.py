import pickle
import numpy as np 
import pandas as pd 
import sklearn as sk 
import flask as fl 
import io
import base64
import heapq
from PIL import Image
import skimage as si
app = fl.Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def begin():
    return fl.render_template("index.html") 
@app.route('/predict/',methods=['GET','POST'])
def predict():
    img = Image.open(fl.request.files['Image'])
    img_data = si.transform.resize(si.io.imread(fl.request.files['Image']),(100,100,3))
    img_data = img_data.flatten()
    prob = model.predict_proba([img_data])
    l = sorted(zip(prob[0],model.classes_),reverse = True)
    Tr = model.predict([img_data])
    print(l)
    mem = io.BytesIO()
    img.save(mem,"JPEG")
    enc_img = base64.b64encode(mem.getvalue())
    return fl.render_template("index.html",img = enc_img.decode('utf-8'),First = l[0][1],FirstP = l[0][0]*100, Second = l[1][1], SecondP = l[1][0]*100)
app.run()