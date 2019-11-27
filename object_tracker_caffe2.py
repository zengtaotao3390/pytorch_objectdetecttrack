from utils import *
from utils import utils

import os, sys, time, datetime, random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
from PIL import Image
from det import  det_model

# load weights and set defaults
img_size=416
conf_thres=0.8
nms_thres=0.4

config_file = './config/head_detect_1gpu_e2e_faster_rcnn_R-50-FPN_2x.yaml'
weights_file = './config/model_iter99999.aug.pkl'
test_img = './data/head.jpg'

m_det = det_model(config_file, weights_file, 1)



videopath = './data/MOV_0845.mp4'


from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

# cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,29.0,(vw,vh))

frames = 0
starttime = time.time()
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = m_det.infer(frame)


    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    if detections is not None:
        tracked_objects = mot_tracker.update(detections)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            color = colors[int(obj_id) % len(colors)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(frame, str(int(obj_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    # cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
print(totaltime)
cv2.destroyAllWindows()
outvideo.release()
