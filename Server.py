from flask import Flask, request, jsonify, Response
from ObjectTrack import ObjectTrack
import numpy as np
import celery_app.task
from Tracker import Tracker
app = Flask(__name__)
import base64


objectTracker = None
tracker = None

@app.route('/track', methods=['POST'])
def track():
    global objectTracker
    if objectTracker is None:
        objectTracker = ObjectTrack()
    file = request.files['file']
    imageArray = np.asarray(bytearray(file.read()), dtype='uint8')
    trackMsg = objectTracker.tracker(imageArray)
    return (jsonify(trackMsg), 200)


@app.route('/trackHead', methods=['POST'])
def trackHead():
    global objectTracker
    if objectTracker is None:
        objectTracker = ObjectTrack()
    file = request.files['file']
    imageArray = np.asarray(bytearray(file.read()), dtype='uint8')
    print(imageArray.shape)
    trackMsg = objectTracker.traceHead(imageArray)
    return (jsonify(trackMsg), 200)


@app.route('/trackHeadTask', methods=['POST'])
def trackHeadTask():
    global tracker
    if tracker is None:
        tracker = Tracker()
    file = request.files['file']
    # imageBytes = base64.b64encode(file.read())
    try:
        result_exc = celery_app.task.headsDet.s(file.read()).delay()
        heads = result_exc.get(timeout=15)
        if heads is None:
            return (jsonify([]), 200)
        heads1 = np.ones((len(heads), 7))
        for i in range(len(heads)):
            head = heads[i]
            for j in range(len(head)):
                heads1[i, j] = head[j]
        trackMsg = tracker.traceHead(heads1)
    except Exception as e:
        trackMsg=[]
    return (jsonify(trackMsg), 200)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6006)