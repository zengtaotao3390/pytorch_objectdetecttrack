from models import *
from utils import utils
import torch
import cv2
from sort import *
from torchvision import transforms
import numpy as np
from det import det_model

class Tracker():
    def __init__(self):
        self.img_size = 416
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.horizontalDividingLine = 800
        self.personIn = 0
        self.personOut = 0
        self.heads = dict()
        self.mot_tracker = Sort()


        print('init finished')



    def traceHead(self, detections):
        trackerMsgs = []
        if detections is not None:
            tracked_objects = self.mot_tracker.update(detections)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                width = int(x2 - x1)
                height = int(y2 - y1)
                obj_id = int(obj_id)
                trackerMsg = {
                    'left': x1,
                    'width': width,
                    'top': y1,
                    'height': height,
                    'objectId': obj_id,
                    'class': int(cls_pred)
                }
                headCenter = width / 2 + x1
                if obj_id not in self.heads:
                    self.heads[obj_id] = headCenter
                else:
                    if self.heads[obj_id] <= self.horizontalDividingLine < headCenter:
                        self.personOut += 1
                        self.heads.pop(obj_id)
                    elif headCenter <= self.horizontalDividingLine < self.heads[obj_id]:
                        self.personIn += 1
                        self.heads.pop(obj_id)
                trackerMsg['personIn'] = self.personIn
                trackerMsg['personOut'] = self.personOut
                trackerMsgs.append(trackerMsg)
            return trackerMsgs

