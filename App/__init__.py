from flask import Flask
from flask_cors import CORS
import os
app = Flask(__name__)
CORS (app) 
app.secret_key='face_prediction'
app.config["MAX_CONTENT_LENGTH"]=4*1024*1024
BASE_DIR= os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR= os.path.join(BASE_DIR,'static','Images' )

import  App.routes 



