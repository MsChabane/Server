from App import app

#from App.face_recognition import Face_Recognition
from flask import request,render_template,jsonify


@app.route('/',methods=["Get","POST"])
def home():
    return "prediction"


@app.route('/face_prediction',methods=['POST'])
def predir():
    img = request.files.get('file')
    
    
    return jsonify({
        "image ": type(img)
    })



