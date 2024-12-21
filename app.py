from flask import Flask,jsonify,request


from flask_cors import CORS
 

class Pca_norme :
  def __init__(self,seuil) -> None:
    self.seuil=seuil
    self.standardScaler=StandardScaler()

  def fit(self ,X):
    self.standardScaler.fit(X)
    cor =  np.corrcoef(self.standardScaler.transform(X).T)
    self.valeurs_propre,self.vecteurs_propre = np.linalg.eig(cor)
    sort_arg = self.valeurs_propre.argsort()[::-1]
    self.valeurs_propre=self.valeurs_propre[sort_arg]
    self.vecteurs_propre=self.vecteurs_propre[:,sort_arg]
    number_of_axes=0
    for i in range(len(self.valeurs_propre)):
      if np.sum(self.valeurs_propre[:i+1])/np.sum(self.valeurs_propre)>=self.seuil:
        number_of_axes=i
        break
    self.valeurs_propre=self.valeurs_propre[:number_of_axes+1]
    self.vecteurs_propre=self.vecteurs_propre[:,:number_of_axes+1] 
  def transform(self,X):
    return self.standardScaler.transform(X).dot(self.vecteurs_propre)

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
   