import cv2

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils
from detectron.utils.vis import convert_from_cls_format as cvt_cls

import time
from collections import defaultdict
from detectron.utils.timer import Timer
import numpy as np

class det_model(object):
    
    def __init__( self, config_file, weights_file, gpu_id ):
        c2_utils.import_detectron_ops()
        
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=3'])
    
        merge_cfg_from_file( config_file )
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg(cache_urls=False)
        self.__model = infer_engine.initialize_model_from_cfg( weights_file, gpu_id=gpu_id )
        self.gpu_id = gpu_id
    
    def infer( self, im, thresh=0.5 ):
        
        ts = time.time()
        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope( self.gpu_id ):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.__model, im, None, timers=timers )
        
        te = time.time()
        print( 'det time:', te-ts )
        for k, v in timers.items():
            print(' | {}: {:.3f}s'.format(k, v.average_time))
        
        if isinstance(cls_boxes, list):
            boxes, segms, keypoints, classes = cvt_cls(
                cls_boxes, cls_segms, cls_keyps)

        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
            return None
        heads = list()
        for i in range( len(boxes) ):
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < thresh:
                continue
            heads.append( [ int(x) for x in bbox ] + [float(score)])
        return heads
        # heads1 = np.ones((len(heads), 7))
        # for i in range( len(heads) ):
        #     head = heads[i]
        #     for j in range(len(head)):
        #         heads1[i, j] = head[j]
        # return heads1

if __name__ == '__main__':
    config_file = './config/head_detect_1gpu_e2e_faster_rcnn_R-50-FPN_2x.yaml'
    weights_file = './config/model_iter99999.aug.pkl'
    test_img = './data/head.jpg'
    
    m_det = det_model( config_file, weights_file, 1)
    
    img = cv2.imread( test_img )
    res = m_det.infer( img )
    
    for b in res:
        cv2.rectangle( img, (b[0], b[1]), (b[2], b[3]), (0,255,0),thickness=2 )
    print( res )
    
    cv2.imwrite( './data/res.jpg', img )