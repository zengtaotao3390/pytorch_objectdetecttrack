# !/usr/bin/env python
# -*- coding: utf-8 -*-
from det import det_model



class chefshat_manager():
    
    def __init__( self, det_cfg, det_wts, gpu_id ):
        self.__m_det = det_model( det_cfg, det_wts, gpu_id )

    def infer( self, img):
        heads = self.__m_det.infer(img)
        return heads
