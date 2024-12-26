from App import app

import pickle
from App.face_recognition import Face_Recognition
from flask import request,jsonify
import base64
import io  

with open ('training.pickle','rb') as f :
    training = pickle.load(f)

with open ("testing.pickle",'rb') as f :
    testing=pickle.load(f)

face = Face_Recognition(training)


@app.route('/',methods=["GET","POST"])
def home():
    return jsonify({
        "name":"App for face recognition",
        "version":"1.0.0",
        "data":{
            "images":407,
            "size":"64 X 64",
            "train":284,
            "test":123,
            "KNN_neihbord":3,
            'threshold of pca ':0.8
        }
    })

@app.route("/face_prediction/make_test/<id>",methods=["POST"])
def handel_testing(id):
    try:
        id = int(id)
        prediction= face.predict(testing[id,:-1].reshape((1,-1)))
        return jsonify({
            "real":str(testing[id,-1]),
            "prediction":str(prediction[0]),
        })
    except :
        return jsonify({
            "erreur":"must give the number of inscance"
        })
     

@app.route("/face_prediction/prediction_source",methods=["POST"])
def handel():
    try:
        data = request.json
        base64_image = data.get('image')
        header,body = base64_image.split(",")
        
        image_data = base64.b64decode(body)
        
        prediction = face.predict_source(io.BytesIO(image_data))
        return jsonify({
            "faceid" :str(prediction[0]),
        })
    except Exception as err:
        res=  jsonify ({
            "erreur":str(err)
        })
        res.status_code =400
        return res