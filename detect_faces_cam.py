import os

import numpy as np
from PIL import Image
import cv2
import torch
from mtcnn import MTCNN
from mtcnn_pytorch.src.visualization_utils import show_bboxes

from retinaface.RetinaFace import RetinaModel

from videostream.videostream import QueuedStream




if __name__ == "__main__":

    save_dir = "image"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    face_id = 0
    
    extend = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #mtcnn = MTCNN()
    retina = RetinaModel()

    stream = QueuedStream(uri=[], drop=True, fps=25)

    stream.start()

    while True:
        ret, frame, frame_id = stream.read()

        if not ret:
            break
        
        resp = retina.detect_faces(frame)
        
        faces = retina.extract_faces(frame, obj=resp)
        
        # image = Image.fromarray(np.uint8(frame)).convert('RGB')
        # bboxes, landmarks = mtcnn.detect_faces(image)
        
        # bboxes, faces = mtcnn.align_intermediate(img=image, boxes=bboxes, landmarks=landmarks)
        
        for face in faces:
            face_id += 1
            face_img = np.asarray(face)
            cv2.imwrite(os.path.join(save_dir, "face_{}.jpg".format(face_id)), face_img)
        
        # #print(bboxes)
        # draw_img = show_bboxes(img=image, bounding_boxes=bboxes, facial_landmarks=landmarks)
        # frame = np.asarray(draw_img)
        
        cv2.imshow("", frame)
        cv2.waitKey(1)

        #print(stream.fps)