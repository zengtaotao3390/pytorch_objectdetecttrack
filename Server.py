from flask import Flask, request, jsonify, Response
from ObjectTrack import ObjectTrack
import numpy as np

app = Flask(__name__)
objectTracker = ObjectTrack()

@app.route('/track', methods=['POST'])
def track():
    file = request.files['file']
    imageArray = np.asarray(bytearray(file.read()), dtype='uint8')
    trackMsg = objectTracker.tracker(imageArray)
    return (jsonify(trackMsg), 200)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6789)