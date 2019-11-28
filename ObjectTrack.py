from models import *
from utils import utils
import torch
import cv2
from sort import *
from torchvision import transforms
import numpy as np
from det import det_model

class ObjectTrack():
    def __init__(self):
        # load weights and set defaults
        self.config_path = 'config/yolov3.cfg'
        self.weights_path = 'config/yolov3.weights'
        self.class_path = 'config/coco.names'
        self.img_size = 416
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.horizontalDividingLine = 800
        self.personIn = 0
        self.personOut = 0

        # load model and put into eval mode
        self.model = Darknet(self.config_path, img_size=self.img_size)
        self.model.load_weights(self.weights_path)
        self.model.cuda()

        self.model.eval()

        self.classes = utils.load_classes(self.class_path)
        self.Tensor = torch.cuda.FloatTensor
        # self.Tensor = torch.FloatTensor
        self.mot_tracker = Sort()
        self.classes = utils.load_classes(self.class_path)
        self.persons = dict()
        self.heads = dict()
        config_file = './config/head_detect_1gpu_e2e_faster_rcnn_R-50-FPN_2x.yaml'
        weights_file = './config/model_iter99999.aug.pkl'
        self.m_det = det_model(config_file, weights_file, 1)


        print('init finished')

    def detect_image(self, img):
        # scale and pad image
        ratio = min(self.img_size / img.size[0], self.img_size / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                             transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                             max(int((imh - imw) / 2), 0),
                                                             max(int((imw - imh) / 2), 0)),
                                                            (128, 128, 128)),
                                             transforms.ToTensor(),
                                             ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(self.Tensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
        return detections[0]

    def tracker(self, imgArray):
        img = cv2.imdecode(imgArray, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilImg = Image.fromarray(img)
        detections = self.detect_image(pilImg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
        unpad_h = self.img_size - pad_y
        unpad_w = self.img_size - pad_x
        trackerMsgs = []
        if detections is not None:
            tracked_objects = self.mot_tracker.update(detections.cpu())
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                cls = self.classes[int(cls_pred)]
                obj_id = int(obj_id)
                trackerMsg = {
                    'left': x1,
                    'width': box_w,
                    'top': y1,
                    'height': box_h,
                    'class': cls,
                    'objectId': obj_id
                }
                if cls == 'person':
                    personCenter = box_w / 2 + x1
                    if obj_id not in self.persons:
                        # 如果是横向移动，横向中心
                        self.persons[obj_id] = personCenter
                    else:
                        if self.persons[obj_id] <= self.horizontalDividingLine < personCenter:
                            self.personOut += 1
                            self.persons.pop(obj_id)
                        elif personCenter <= self.horizontalDividingLine < self.persons[obj_id]:
                            self.personIn += 1
                            self.persons.pop(obj_id)
                trackerMsg['personIn'] = self.personIn
                trackerMsg['personOut'] = self.personOut
                trackerMsgs.append(trackerMsg)
        return trackerMsgs


    def traceHead(self, imgArray):
        img = cv2.imdecode(imgArray, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.m_det.infer(img)
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

if __name__ == '__main__':
    track = ObjectTrack()
    image = Image.open('images/blueangels.jpg')
    print(track.tracker(np.asarray(image)))
