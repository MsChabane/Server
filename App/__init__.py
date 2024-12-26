from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS (app) 
app.secret_key='face_prediction'
app.config["MAX_CONTENT_LENGTH"]=4*1024*1024


import  App.routes 



