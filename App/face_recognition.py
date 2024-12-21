import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  PIL import Image





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