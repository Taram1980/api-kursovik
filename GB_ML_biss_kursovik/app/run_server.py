# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import json

import dill
from tqdm import tqdm
import numpy as np  
import datetime
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime


# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(modelpath, 'rb') as f:
		model = dill.load(f)
	print(model)


modelpath = r'./models/logreg_pipeline3.dill'
load_model(modelpath)

@app.route("/", methods=["GET"])
def general():
	return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	if flask.request.method == "POST":
		request_json = pd.read_json(flask.request.get_json())
		try:
			preds = model.predict(request_json)
		except:
			preds = ['error']
		data["predictions"] = preds
		data["success"] = True
	return flask.jsonify(pd.Series(data).to_json(orient='values'))

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='127.0.0.1', debug=True, port=port)
