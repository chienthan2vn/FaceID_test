import cv2 
import numpy as np
from ultralytics import YOLO
from architecture import *
from train_v2 import normalize,l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle

recognition_t = 0.15
required_size = (160,160)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img ,detector,encoder,encoding_dict):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = 0, 0, 0, 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = detector(img_rgb)
    boxes = x[0].boxes
    for box in boxes:
        top_left_x = int(box.xyxy.tolist()[0] [0])
        top_left_y= int(box.xyxy.tolist() [0] [1])
        bottom_right_x = int(box.xyxy.tolist() [0] [2])
        bottom_right_y= int(box.xyxy.tolist() [0] [3])
        face = img_rgb[top_left_y:bottom_right_y , top_left_x:bottom_right_x]

        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)
            cv2.putText(img, name, (top_left_x, top_left_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (top_left_x, top_left_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img 



if __name__ == "__main__":
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path = "facenet_keras_weights.h5"
    face_encoder.load_weights(path)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = YOLO("./model/best.pt")
    encoding_dict = load_pickle(encodings_path)
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            print("CAM NOT OPEND") 
            break
        
        frame= detect(frame , face_detector , face_encoder , encoding_dict)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    


