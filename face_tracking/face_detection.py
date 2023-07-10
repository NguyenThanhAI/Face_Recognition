import numpy as np


class FaceDetection(object):
    def __init__(self,bbox, detection_id):
        self.bbox = bbox
        self.detection_id = detection_id