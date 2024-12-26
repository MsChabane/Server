import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  PIL import Image

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
  
  
class Face_Recognition :
  def __init__(self,data,seuil=0.8,neighbors=3) -> None:
    if data is None :
      raise ValueError("data is undefined")
    
    if seuil == None or seuil <=0 or seuil >1 :
      raise ValueError("seuil is invalid")
    if neighbors == 0 or not neighbors  :
      raise ValueError("neighbors is invalid")
    self.stadardScaler = StandardScaler().fit(data[:,:-1])
    self.acp=PCA(n_components=int(data.shape[0]*seuil))
    self.Knn=KNeighborsClassifier(n_neighbors=neighbors)
    self._fit(data[:,:-1],data[:,-1])
    
  def _fit(self,X,Y):
      X_scaled = self.stadardScaler.transform(X)
      X_scaled=self.acp.fit_transform(X_scaled)
      self.Knn.fit(X_scaled,Y)

  def _make_prediction(self,X):
      X_scaled=self.stadardScaler.transform(X)
      X_scaled=self.acp.transform(X_scaled)
      return self.Knn.predict(X_scaled)
  
  def predict(self ,X_test):
      return self._make_prediction(X_test)
     
  def predict_source(self,image_uri):
      image =  np.array( Image.open(image_uri).resize((64,64)).convert("L")).reshape((1,-1))
      return self._make_prediction(image)