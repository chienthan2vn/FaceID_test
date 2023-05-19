from architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from ultralytics import YOLO
from tensorflow.keras.models import load_model

######pathsandvairables#########
face_data = 'Faces/'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_detector = YOLO("./model/best.pt")
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data,face_names)

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir,image_name)

        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        x = face_detector(img_RGB)
        boxes = x[0].boxes
        for box in boxes:
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y= int(box.xyxy.tolist() [0] [1])
            bottom_right_x = int(box.xyxy.tolist() [0] [2])
            bottom_right_y= int(box.xyxy.tolist() [0] [3])
            face = img_RGB[top_left_y:bottom_right_y , top_left_x:bottom_right_x]
            # cv2.rectangle(img_RGB, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (50, 200, 129), 2)

            face = normalize(face)
            face = cv2.resize(face, required_shape)
            face_d = np.expand_dims(face, axis=0)
            encode = face_encoder.predict(face_d)[0]
            print("encode predict: ", encode.shape)
            encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0 )
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        # print("encode normal: ", encode)
        encoding_dict[face_names] = encode
    
path = 'encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)






