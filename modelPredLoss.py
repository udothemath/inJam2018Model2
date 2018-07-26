import numpy as np
from flask import Flask, abort, jsonify, request
import pickle

myPickle = 'model2.pkl'

modelCatBoostRegressor = pickle.load(open(myPickle, 'rb'))

app = Flask(__name__)

@app.route('/model2', methods=['POST'])
def make_predict():
	
	## all kinds of error checking should go here
	data = request.get_json(force=True)
	## convert our json to a numpy array
	predict_request = [data['Group'], 
		               data['CnaeSession'], 
                       data['Uf'], 
                       data['FundationDate'], 
                       data['Equity']]
	predict_request = np.array(predict_request)
	predict_reshape = np.array(predict_request).reshape(1, -1)

	predLoss = modelCatBoostRegressor.predict(predict_reshape)
	#print(predLoss)

	## return our prediction
	return jsonify(predLoss = predLoss.tolist())


if __name__ == '__main__':
    app.run(port = 8000, debug = True)
