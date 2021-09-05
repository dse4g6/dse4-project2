from flask import Flask, request, jsonify, make_response
import predict_social as social
import predict_mango as mango
import random

import pandas as pd
import gspread

import requests




app = Flask(__name__)

@app.route('/', methods=["GET"])
def hello_world():
    return 'Hello, DSE#4 Group66'

@app.route('/socialtime')
def socialMedia ():
    gender = request.args.get('gender')
    age = request.args.get('age')
    mobile = request.args.get('mobile')
    relationship = request.args.get('relationship')

    # gender = 'ชาย'
    # age = '21'
    # mobile = 'Android'
    # relationship = 'มี'
    

    answer_list = [gender, age, mobile, relationship]
    pred = social.pred_pipeline(answer_list)
    pred = round(pred[0], 2)

    result = {'res': pred}

    return jsonify(result)

@app.route('/mango-prediction', methods=["GET"])
def mango_prediction():
    url = request.args.get("url")
    # mango_list = ["เขียวเสวย", "ฟ้าลั่น", "แรด", "มันขุนศรี"]

    pred = mango.pred_pipeline(url)
    pred = pred[0]

    jsonData = {
        "url" : url,
        # "result" : random.choice(mango_list),
        "result" : pred
    }
    return jsonify(jsonData)


@app.route('/post', methods=["POST"])
def post():
    return 'Hello World Post'


@app.route('/query', methods=["POST"])
def query():
    # print(request.args)
    name = request.args.get("name")
    return 'Hello ' + name

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
    # app.run(host="0.0.0.0", debug=False)
