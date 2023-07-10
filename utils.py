import os
from typing import List

import math

from PIL import Image

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


def face_orientation(frame: np.ndarray, landmark):
    size = frame.shape
    
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])
    
    image_points = np.array([
                            (landmark[30][0], landmark[30][1]),     # Nose tip
                            (landmark[8][0], landmark[8][1]),   # Chin
                            (landmark[45][0], landmark[45][1]),     # Left eye left corner
                            (landmark[36][0], landmark[36][1]),     # Right eye right corne
                            (landmark[48][0], landmark[48][1]),     # Left Mouth corner
                            (landmark[54][0], landmark[54][1])      # Right mouth corner
                        ], dtype="double")
    
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    
    return roll, pitch, yaw


def compute_head_pose(landmark):
    # Facial landmark indices for relevant points
    left_eye_indices = list(range(36, 42))
    right_eye_indices = list(range(42, 48))
    nose_tip_index = 30
    mouth_center_indices = [48, 54]

    # Calculate eye centers
    left_eye_center = np.mean(landmark[left_eye_indices], axis=0)
    right_eye_center = np.mean(landmark[right_eye_indices], axis=0)

    # Compute roll angle
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    roll_angle = math.atan2(dy, dx) * 180.0 / math.pi

    # Compute pitch angle
    dx = left_eye_center[0] - landmark[nose_tip_index][0]
    dy = left_eye_center[1] - landmark[nose_tip_index][1]
    pitch_angle = math.atan2(dy, dx) * 180.0 / math.pi

    # Compute yaw angle
    dx = left_eye_center[0] - landmark[mouth_center_indices[0]][0]
    dy = left_eye_center[1] - landmark[mouth_center_indices[0]][1]
    yaw_angle = math.atan2(dy, dx) * 180.0 / math.pi

    return roll_angle, pitch_angle, yaw_angle


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

#this function copied from the deepface repository: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
def alignment_procedure(img, left_eye, right_eye, nose):

    #this function aligns given face in img based on left and right eye coordinates

    #left eye is the eye appearing on the left (right eye of the person)
    #left top point is (0, 0)

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    #-----------------------
    #decide the image is inverse

    center_eyes = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))
    
    if False:

        img = cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 2, (0, 255, 255), 2)
        img = cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 2, (255, 0, 0), 2)
        img = cv2.circle(img, center_eyes, 2, (0, 0, 255), 2)
        img = cv2.circle(img, (int(nose[0]), int(nose[1])), 2, (255, 255, 255), 2)

    #-----------------------
    #find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock

    #-----------------------
    #find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    #-----------------------

    #apply cosine rule

    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        
        #PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
        #In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))
        
        
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree

        #-----------------------
        #rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle).resize(size=(112, 112)))

    #-----------------------

    return img #return img anyway