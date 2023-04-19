import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('CVGR_Data')
actions = np.array(['background', 'yes', 'no', 'milk', 'coffee', 'small', 'large'])
num_sequences = 30
sequence_length = 30

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[result.x, result.y, result.z, result.visibility] for result in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[result.x, result.y, result.z, result.visibility] for result in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    leftHand = np.array([[result.x, result.y, result.z, result.visibility] for result in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rightHand = np.array([[result.x, result.y, result.z, result.visibility] for result in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, leftHand,rightHand])



cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(num_sequences):
            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                image, results = mediapipe_detection(frame, holistic)
                print(results)
                
                draw_landmarks(image, results)

                if frame_num==0:
                    cv2.putText(image, 'Data Collection Starting', (120,200),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),4,cv2.LINE_AA)
                    cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action,sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.waitKey(3000)
                else:
                    cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action,sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                cv2.imshow('camera feed', image)

                if cv2.waitKey(10) & 0xFF == 27:
                    break

                cap.release()
                cv2.destroyAllWindows()


result_test=extract_keypoints(results)
np.save('0', result_test)


print(result_test)
