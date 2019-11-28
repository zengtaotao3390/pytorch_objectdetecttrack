from flask import Flask, request, jsonify, Response
import numpy as np
app = Flask(__name__)
import base64
import matplotlib.pyplot as plt
import cv2


@app.route('/trackHeadTask', methods=['POST'])
def trackHeadTask():
    file = request.files['file']
    imageStr = base64.b64encode(file.read())
    imageDict = {'image': imageStr}
    print(imageDict)
    imgByte = base64.b64decode(imageStr)
    npArray = np.asarray(bytearray(imgByte))
    img = cv2.imdecode(npArray, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    plt.imshow(img)
    plt.show()
    return (jsonify(), 200)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6006)