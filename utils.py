import os
from typing import List

import numpy as np
import cv2


def enumerate_images(data_dir: str) -> List[str]:
    file_list = []
    for dirs, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".jpg", ".png")):
                file_list.append(os.path.join(dirs, file))
                
    return file_list


def preprocess(img: np.ndarray):
    if not isinstance(img, np.ndarray):
        img = cv2.imread(img)
    if img.shape[0] != 112 or img.shape[1] != 112:
        img = cv2.resize(img, (112, 112))
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    
    return img