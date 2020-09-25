import random
import os
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service
import train
import prepare_dataset


# instantiate flask app
app = Flask(__name__)
DATASET_PATH = "dataset"
JSON_PATH = "data.json"

@app.route("/label", methods =["POST"])
def label():
	file = request.files["file"]
	label = request.form['label']

	if(os.path.isdir("dataset\\"+label)):
		print("Word Already Exists")
	else:
		print("Word Does not exist")
		os.mkdir("dataset\\"+label)
		newDirectory = "dataset\\"+label
		file.save(os.path.join(newDirectory,file.filename))
		print("=-=-=-=-=-=-= \n DONE TRAINING")


	prepare_dataset.prepare_dataset(DATASET_PATH, JSON_PATH)
	train.main()

	return jsonify({})






@app.route("/predict", methods=["POST"])
def predict():
	"""Endpoint to predict keyword
    :return (json): This endpoint returns a json file with the following format:
        {
            "keyword": "down"
        }
	"""

	# get file from POST request and save it
	audio_file = request.files["file"]
	file_name = str(random.randint(0, 100000))
	audio_file.save(file_name)

	# instantiate keyword spotting service singleton and get prediction
	kss = Keyword_Spotting_Service()
	predicted_keyword = kss.predict(file_name)

	# we don't need the audio file any more - let's delete it!
	os.remove(file_name)

	# send back result as a json file
	result = {"keyword": predicted_keyword.split("\\")[-1]}
	return jsonify(result)


if __name__ == "__main__":
    app.run(host = '0.0.0.0',port="5000",debug=False)
