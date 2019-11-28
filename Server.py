from flask import Flask, request, jsonify, Response
# from ObjectTrack import ObjectTrack
import numpy as np
# import celery_app.task
# from Tracker import Tracker
app = Flask(__name__)
import base64
import matplotlib.pyplot as plt
import cv2
objectTracker = None
tracker = None
import PIL.Image as Image

# @app.route('/track', methods=['POST'])
# def track():
#     global objectTracker
#     if objectTracker is None:
#         objectTracker = ObjectTrack()
#     file = request.files['file']
#     imageArray = np.asarray(bytearray(file.read()), dtype='uint8')
#     trackMsg = objectTracker.tracker(imageArray)
#     return (jsonify(trackMsg), 200)


# @app.route('/trackHead', methods=['POST'])
# def trackHead():
#     global objectTracker
#     if objectTracker is None:
#         objectTracker = ObjectTrack()
#     file = request.files['file']
#     imageArray = np.asarray(bytearray(file.read()), dtype='uint8')
#     print(imageArray.shape)
#     trackMsg = objectTracker.traceHead(imageArray)
#     return (jsonify(trackMsg), 200)


@app.route('/trackHeadTask', methods=['POST'])
def trackHeadTask():
    # global tracker
    # if tracker is None:
    #     tracker = Tracker()
    file = request.files['file']
    imageArray = np.asarray(bytearray(file.read()), dtype='uint8')
    # imageStr = base64.b64encode(file.read())
    # imgByte = base64.b64decode(imageStr)
    npArray = np.asarray(bytearray(imageArray))
    img = cv2.imdecode(npArray, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    plt.imshow(img)
    plt.show()
    # print(imageStr)
    # imageDic = {"imageStr": imageStr}
    # result_exc = celery_app.task.headsDet.s(file).delay()
    # dict_exc = result_exc.get(timeout=15)
    # trackMsg = tracker.traceHead(dict_exc)
    return (jsonify(), 200)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6006)