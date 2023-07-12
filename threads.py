import os
import numpy as np
import cv2
from PyQt5 import QtCore, QtWidgets
from singletons import Detector, RecognitionModel
from face_tracking.face_tracker import FaceTracker
from face_tracking.face_detection import FaceDetection
from videostream.videostream import QueuedStream


class DefaultProcess(QtCore.QThread):
    update_frame = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, network, weights) -> None:
        super(DefaultProcess, self).__init__()
        self.network = network
        self.weights = weights
        self.stopped = False
        
    def run(self):
        stream = QueuedStream(uri=[], drop=True, fps=25)
        detector = Detector()
        face_recog_model = RecognitionModel(network=self.network, weights=self.weights)
        tracker = FaceTracker(inactive_steps_before_removed=20,
                              reid_iou_threshold=0.6,
                              max_traject_steps=30,
                              tentative_steps_before_accepted=3)
        stream.start()
        
        if not stream.isOpened():
            print("Can not open video")
            return
        
        
        while not self.stopped:
            ret, frame, frame_id = stream.read()
            if not ret:
                break
            
            dets = detector.get_landmarks_from_image(frame, return_bboxes=True, return_landmark_score=True)
            bboxes = dets[2]
            landmarks = dets[0]
            
            
            try:
                
                detections_list = []
                for i, (landmark, bbox) in enumerate(zip(landmarks, bboxes)):
            
                    x_1, y_1, x_2, y_2, prob = bbox
                    if prob > 0.7:
                        w = x_2 - x_1
                        h = y_2 - y_1

                        detections_list.append(FaceDetection(bbox=bbox, landmark=landmark, detection_id=i))
                
                tracker.step(face_detections=detections_list)
                face_tracks = tracker.get_result()
                for face_track in face_tracks:
                    bbox = face_track.bbox
                    landmark = face_track.landmark
                    #print(bbox)
                    track_id = face_track.track_id
                    x_min, y_min, x_max, y_max, prob = bbox
                    result = "Track id: {}".format(track_id)
                    x_min = int(x_min)
                    x_max = int(x_max)
                    y_min = int(y_min)
                    y_max = int(y_max)   
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
                    cv2.putText(frame, result, (x_min + int(0.05*(x_max - x_min)), y_min - int(0.05*(y_max - y_min))), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)
                    
                    for j, point in enumerate(landmark):
                        if j in [8, 30, 36, 45, 48, 54]:
                            # 8: Cằm
                            # 30: Mũi
                            # 36: Mắt trái
                            # 45: Mắt phải
                            # 48: Miệng trái
                            # 54: Miệng phải
                            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                            cv2.putText(frame, str(j), (int(point[0]) + 3, int(point[1]) - 3),  cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)
                    
                self.update_frame.emit(frame)
            except Exception as e:
                print("Exception: {}".format(e))
                continue