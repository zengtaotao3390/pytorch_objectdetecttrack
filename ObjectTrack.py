from models import *
from utils import utils
import torch
import cv2
from sort import *
from torchvision import transforms
import numpy as np

class ObjectTrack():
    def __init__(self):
        # load weights and set defaults
        self.config_path = 'config/yolov3.cfg'
        self.weights_path = 'config/yolov3.weights'
        self.class_path = 'config/coco.names'
        self.img_size = 416
        self.conf_thres = 0.8
        self.nms_thres = 0.4

        # load model and put into eval mode
        self.model = Darknet(self.config_path, img_size=self.img_size)
        self.model.load_weights(self.weights_path)
        # self.model.cuda()

        self.model.eval()

        self.classes = utils.load_classes(self.class_path)
        # self.Tensor = torch.cuda.FloatTensor
        self.Tensor = torch.FloatTensor
        self.mot_tracker = Sort()
        self.classes = utils.load_classes(self.class_path)

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

    def tracker(self, imageArray):
        pilImg = Image.fromarray(imageArray)
        detections = self.detect_image(pilImg)
        pad_x = max(imageArray.shape[0] - imageArray.shape[1], 0) * (self.img_size / max(imageArray.shape))
        pad_y = max(imageArray.shape[1] - imageArray.shape[0], 0) * (self.img_size / max(imageArray.shape))
        unpad_h = self.img_size - pad_y
        unpad_w = self.img_size - pad_x
        trackerMsgs = []
        if detections is not None:
            trackerMsg = {}
            tracked_objects = self.mot_tracker.update(detections.cpu())
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * imageArray.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * imageArray.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * imageArray.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * imageArray.shape[1])
                cls = self.classes[int(cls_pred)]
                trackerMsg = {
                    'x1': x1,
                    'x2': x1+box_w,
                    'y1': y1,
                    'y2': y1+box_h,
                    'class': cls,
                    'objectId': int(obj_id)
                }
                trackerMsgs.append(trackerMsg)
        return trackerMsgs


if __name__ == '__main__':
    track = ObjectTrack()
    image = Image.open('images/blueangels.jpg')
    track.tracker(np.asarray(image))
