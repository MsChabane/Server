from App import app,IMAGES_DIR
from PIL import Image 
import pickle
from App.face_recognition import Face_Recognition
from flask import request,jsonify
import base64
import io 
import os  

with open ('training.pickle','rb') as f :
    training = pickle.load(f)

with open ("testing.pickle",'rb') as f :
    testing=pickle.load(f)
    
with open ("target_data.pickle",'rb') as f :
    target_data=pickle.load(f)
    
    

face = Face_Recognition(training)


@app.route("/face_prediction/get_persones")
def get():
    data =[]
    images = os.listdir(IMAGES_DIR)
    
    for idx in range(len(images)):
        try :
            faceid =int(images[idx].split(".")[0])
            img_path =os.path.join(IMAGES_DIR,images[idx])
            with Image.open(img_path) as img :
                buffer =io.BytesIO()
                img.save(buffer,format="PNG")
                img_base64=base64.b64encode(buffer.getvalue()).decode("utf-8")
                data.append({
                    "id":faceid,
                    "name":target_data[faceid,0],
                    "gender":target_data[faceid,1],
                    "image":img_base64
                })
        except Exception as err:
            return jsonify({
                'erreur':str(err)
            })
            
    return jsonify(data)

@app.route('/',methods=["GET","POST"])
def home():
    return jsonify({
        "name":"App for face prediction",
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
        if id >= 123 or id < 0 : 
            prediction= face.predict(testing[id,:-1].reshape((1,-1)))
            return jsonify({
                "real details":{
                        "id":testing[id,-1],
                        "name":target_data[id,0],
                        "gender":target_data[id,1]
                    },
                "prediction":{
                        "id":prediction[0],
                        "name":target_data[prediction[0],0],
                        "gender":target_data[prediction[0],1]
                    },
            })
        return jsonify({
            "erreur":f"{id} not found" 
        })
    except :
        return jsonify({
            "erreur":f"{id} is not valid (must give number between 0 and 122 to predict the image )"
        })

 

@app.route("/face_prediction/prediction_source",methods=["POST"])
def handel():
    try:
        data = request.json
        base64_image = data.get('image')
        body = base64_image.split(",")[1]
        
        image_data = base64.b64decode(body)
        
        prediction = face.predict_source(io.BytesIO(image_data))
        return jsonify({
            "id":prediction[0],
            "name":target_data[prediction[0],0],
            "gender":target_data[prediction[0],1]
        })
    except Exception as err:
        return  jsonify ({
            "erreur":str(err)
        })
        