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
#  get 15-17 years old pics for teen class 
def get_afadfull(img_path):
    detector = MTCNN()
    agelabel_list = os.listdir(img_path)
    X_age = []
    nonenum = 0
    nofacenum = 0
    bboxnum = 0
    zeronum = 0
    lost = 0

    for agelabel in agelabel_list:
        genderpath = os.path.join(img_path,agelabel)
        genderlabel_list = os.listdir(genderpath)
        for genderlabel in genderlabel_list:
            final_path = os.path.join(genderpath,genderlabel)
            img_list=os.listdir(final_path)
            for name in img_list:
                age = int(agelabel)
                if age >= 0 and age < 12:
                    age = 0  #use age group to cover age
                elif age >= 12 and age < 18:
                    age = 1
                elif age >= 12 and age < 45:
                    age = 2
                elif age >= 45 and age < 65:
                    age = 3
                else:
                    age = 4
                if age == 1:
                    print('{} Age group:{}'.format(name,str(age)))
                    img = cv2.imread(os.path.join(final_path,name))
                    if img is None:
                        nonenum += 1
                        continue
                    result = detector.detect_faces(img)
                    if not result:
                        nofacenum += 1
                        continue
                    face_position = result[0].get('box')
                    x_coordinate = face_position[0]
                    y_coordinate = face_position[1]
                    w_coordinate = face_position[2]
                    h_coordinate = face_position[3]
                    if h_coordinate < 48:
                        print('crop size is smaller than min size 48')
                        bboxnum += 1
                        continue
                    img = img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                    if (img.size == 0):
                        zeronum += 1
                        continue
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (img_size, img_size))
                    X_age.append((img,age))
                    X_age.append((img,age))
    lost = nonenum + nofacenum + bboxnum + zeronum
    print('nonepic:%d, noface:%d, small bbox:%d, zeroface:%d, all lost:%d' % (nonenum,nofacenum,bboxnum,zeronum,lost))
    for _ in range(10):
        np.random.shuffle(X_age)
    print('data size : %d' % (len(X_age)))
    
    np.save('./data/afad-full/' + 'train_age.npy', X_age)
    np.save('./data/afad-full/' + 'test_age.npy', X_age)
    

if __name__ == '__main__':
    get_afadfull(args['image'])



