from flask import Flask, request, jsonify
import predict_social as social

app = Flask(__name__)


@app.route('/', methods=["GET"])
def hello_world():
    return 'Hello, DSE#4 Group6'

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


@app.route("/get_profile", methods=["GET"])
def get_profile():
    name = request.args.get("name")
    jsonData = {
        "fname" : name,
        "lname" : "Pimprasan",
        "nickname" : "Big",
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
