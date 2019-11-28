import sys
sys.path.append('/data/pytorch_objectdetecttrack')
from celery import Celery
from billiard import current_process
import celery.signals
import configparser
from celery.result import AsyncResult
import numpy as np
import base64
import cv2
delattr(AsyncResult, '__del__')

cf = configparser.ConfigParser()
cf.read("config.conf")
worker_concurrency = cf.getint("celery", "worker_concurrency_gpu")
gpu_total = cf.getint("celery", "gpu_total")


app = Celery()
app.config_from_object("celery_app.celeryconfig")

headsDetObject = None

@celery.signals.worker_process_init.connect
def worker_process_init(sender, **kwargs):
    getInstanceClothesAnalysis()


@app.task
def headsDet(imageStr):
    npArray = np.asarray(bytearray(imageStr))
    img = cv2.imdecode(npArray, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_dict = headsDetObject.infer(img)
    return result_dict


def getGpuId():
    index_worker = current_process().index
    gpu_id = index_worker % gpu_total
    return gpu_id

def getInstanceClothesAnalysis():
    global headsDetObject
    config_file = './config/head_detect_1gpu_e2e_faster_rcnn_R-50-FPN_2x.yaml'
    weights_file = './config/model_iter99999.aug.pkl'
    from modelcreator import chefshat_manager
    if headsDetObject == None:
        gpu_id = getGpuId()
        headsDetObject = chefshat_manager(config_file, weights_file, gpu_id)

if __name__ == "__main__":
    app.start()
