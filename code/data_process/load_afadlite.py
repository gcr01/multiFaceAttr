import os
import cv2
import pickle
import numpy as np
import argparse
import re
import csv
from mtcnn.mtcnn import MTCNN
ap=argparse.ArgumentParser()
ap.add_argument('--image',required=True,
                help='Path of images')

args = vars(ap.parse_args())

img_size=48
#1-116岁，0-115 array
def get_afadlite(img_path):
    detector = MTCNN()
    agelabel_list = os.listdir(img_path)
    X_gender = []
    male_count = 0
    female_count = 0
    for agelabel in agelabel_list:
        genderpath = os.path.join(img_path,agelabel)
        genderlabel_list = os.listdir(genderpath)
        for genderlabel in genderlabel_list:
            final_path = os.path.join(genderpath,genderlabel)
            img_list=os.listdir(final_path)
            for name in img_list:
                if genderlabel == '111':
                    label = 1
                if genderlabel == '112':
                    label = -1
                print('{} Gender:{}'.format(name,str(label)))
                img = cv2.imread(os.path.join(final_path,name))
                if img is None:
                    continue
                result = detector.detect_faces(img)
                if not result:
                    continue
                face_position = result[0].get('box')
                x_coordinate = face_position[0]
                y_coordinate = face_position[1]
                w_coordinate = face_position[2]
                h_coordinate = face_position[3]
                if w < 48:
                    print('crop size is smaller than min size 48')
                    continue
                img = img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                if (img.size == 0):
                    continue
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (img_size, img_size))
                if (label == 1) and (male_count<14000):
                    X_gender.append((img,label))
                    male_count += 1
                if (label == -1) and (female_count<14000):
                    X_gender.append((img,label))
                    female_count += 1
    for _ in range(10):
        np.random.shuffle(X_gender)
    print('data size : %d' % (len(X_gender)))
    
    gender_boundary=int( (len(X_gender))*0.9 )
    train_data_gender, test_data_gender = X_gender[:gender_boundary], X_gender[gender_boundary:]
    
    np.save('./data/afad-lite/' + 'train_gender.npy', train_data_gender)
    np.save('./data/afad-lite/' + 'test_gender.npy', test_data_gender)
    

if __name__ == '__main__':
    get_afadlite(args['image'])



