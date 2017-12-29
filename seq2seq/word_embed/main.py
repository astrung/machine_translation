from flask import Flask, request, jsonify, json, Response
import os, requests
# from bson.objectid import ObjectId
# import datetime
from datetime import datetime

app = Flask(__name__)
app.config['DEBUG'] = True
service_host = "127.0.0.1"
service_port = 5000
baseurl = "http://" + service_host + ":" + str(service_port) + "/"

@app.route('/vietnamese', methods=['GET'])
def get_tasks():
    return "Hello"

@app.route('/vietnamese', methods=['POST'])
def new_product():
    parameters = request.get_json(force=True)
    name = parameters["name"]
    return jsonify(name=name)
    # return json.dump({"name": name})

ACCESS_TOKEN = "EAAYSfPA8lnsBALTzt6OFufmytCaEbFw8cnyawNag4ZBcTRyjnAzN36rmlM8GZBvLQLZC4ySEZBPeXjUHXfDEViQRroPsZAmQEYru1wZBB0NO6yQKIvPEeqRm4LZAWNgEp5fQaZA2y8JfVqmjn251T4MDsiwcAK2guhTHAa3GvYILbAZDZD"

# app = Flask(__name__)

@app.route('/', methods=['GET'])
def verify():
    # our endpoint echos back the 'hub.challenge' value specified when we setup the webhook
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == 'foo':
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200
    #
    return 'Hello World (from Flask!)', 200

def reply(user_id, msg):
    data = {
        "recipient": {"id": user_id},
        "message": {"text": msg}
    }
    resp = requests.post("https://graph.facebook.com/v2.6/me/messages?access_token=" + ACCESS_TOKEN, json=data)
    print(resp.content)

@app.route('/', methods=['POST'])
def handle_incoming_messages():
    data = request.json
    sender = data['entry'][0]['messaging'][0]['sender']['id']
    message = data['entry'][0]['messaging'][0]['message']['text']
    print(message)
    reply(sender, message)

    return "ok"

if __name__ == '__main__':
    app.run(host=service_host, port=service_port)
