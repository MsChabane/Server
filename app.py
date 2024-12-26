from flask import Flask,jsonify,request


from flask_cors import CORS
 



class Face_Recognition:
  def __init__(self,data,target,seuil=0.8,neighbors=3) -> None:
    if data is None :
      raise ValueError("data is undefined")
    if target is None :
      raise ValueError("target is undefined")
    if seuil == None or seuil <=0 or seuil >1 :
      raise ValueError("seuil is invalid")
    if neighbors == 0 or not neighbors  :
      raise ValueError("neighbors is invalid")
    self.stadardScaler = StandardScaler().fit(data)
    self.acp=PCA(n_components=int(data.shape[0]*seuil))
    self.Knn=KNeighborsClassifier(n_neighbors=neighbors)
    
    self._fit(data,target)
  
  def _fit(self,X,Y):
      self.acp.fit(X)
      X_scaled = self.stadardScaler.transform(X)
      X_scaled=self.acp.transform(X_scaled)
      self.Knn.fit(X_scaled,Y)

  def predict(self,image_uri:str):
      image =  np.array( Image.open(image_uri).resize((64,64)).convert("L")).reshape((1,-1))
      X_scaled=self.stadardScaler.transform(image)
      X_scaled=self.acp.transform(X_scaled)
      return self.Knn.predict(X_scaled)


app = Flask(__name__)
CORS(app)
#with open("./data.pickle",'rb') as f :
#    data = pickle.load(f)

#with open("./target.pickle",'rb') as f :
#    target = pickle.load(f)

#face =Face_Recognition(data= data ,target=target,seuil=0.8,neighbors=3)

@app.route('/',methods=["Get"])
def home():
    return "prediction"


@app.route('/prediction',methods=["POST","Get"])
def send_user():
    if request.method == "GET":
        return "prediction:"
    try :
       print("test")
       img = request.files.get('file')
       print(type(img))
       print(img)
       imgcode=20
       return jsonify({'image':imgcode })
       
    except Exception as err:
        print(err)
        return jsonify({'err':"erreur " })
   
   
if __name__ =='__main__':
    app.run(debug=True,port=5000)
   