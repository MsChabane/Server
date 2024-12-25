from App import app
import PIL
import pickle
from App.face_recognition import Face_Recognition
from flask import request,render_template,jsonify


with open ('data.pickle','rb') as f :
    data = pickle.load(f)

with open ('target.pickle','rb') as f :
    target = pickle.load(f)  

face = Face_Recognition(data ,target)




@app.route('/',methods=["GET","POST"])
def home():
    return jsonify({
        "name":"App for face recenition",
        "version":"1.0.0"
        
    })


@app.route('/face_prediction',methods=['POST'])
def predir():
    if 'file' in request.files:
        img = request.files.get('file')
        try :
            prediction = face.predict(img)
            return jsonify({
                "Faceid":prediction
            })
        except Exception as err:
            return jsonify ({
                "erreur":err
            })
    return jsonify({
        "erreur":"file not found"
    })
        
@app.route("/test",methods=["GET",'POST']) 
def test():
    try :
        prediction = face.predict("im.jpg")
        return jsonify({
            "Faceid":str(prediction)
        })
        
    except Exception as err:
        print(err)
        return  jsonify ({
            "erreur":"err"
        })


