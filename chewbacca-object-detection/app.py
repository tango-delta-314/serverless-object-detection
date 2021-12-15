from flask import Flask, jsonify, make_response
from os import listdir
import cv2 as cv

app = Flask(__name__)

net = cv.dnn_DetectionModel('/mnt/tapod/yolov4/yolov4.cfg', '/mnt/tapod/yolov4/yolov4.weights')
net.setInputSize(704, 704)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

with open('/mnt/tapod/yolov4/coco.names', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')

@app.route("/")
def hello_from_root():
    dir1 = None
    dir2 = None
    error = None

    try:
        dir1 = listdir()
        dir2 = listdir('/mnt/tapod')
    except Exception as e:
        error = str(e)

    return jsonify(message='Hello from root!', dir1=dir1, dir2=dir2, error=error)


@app.route("/hello")
def hello():

    frame = cv.imread('/mnt/tapod/yolov4/BB-Italia-Defining-Designs.jpg')
    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)

    data = []
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):

        label = "%.2f" % confidence
        label = "%s: %s" % (names[classId], label)

        data.append(label)

    return jsonify(message='Hello from path!', data=data)


@app.errorhandler(404)
def resource_not_found(e):
    return make_response(jsonify(error='Not found!'), 404)
