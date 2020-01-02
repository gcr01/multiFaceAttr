# ---author : xiaoyang---
# coding : utf-8

import pickle
import os
import cv2
import numpy as np
#from mtcnn.mtcnn import MTCNN
import re
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('--image',required=True,
                help='Path of images')
ap.add_argument('--bbox',required=True,
                help='Path of bounding boxes')
ap.add_argument('--attr',required=True,
                help='Path of attribute labels')
args=vars(ap.parse_args())

def get_crop_img(bbox_path,img_path,attr_path): #crop the images from the bounding box in them
    with open(bbox_path,'r') as bbox_file:
        with open(attr_path,'r') as attr_file:
            img_list = os.listdir(img_path)
            img_list.sort()
            num_of_img = len(img_list)
            bbox_list = bbox_file.readlines()
            attr_list = attr_file.readlines()
            X_smile = []
            X_gender = []
            X_glasses = []
            glass_count = 0
            notglass_count = 0
            smile_count = 0
            notsmile_count = 0
            male_count = 0
            female_count = 0
            for i in range(num_of_img):
                img_name = os.path.join(img_path,img_list[i])
                img = cv2.imread(img_name)
                face_bbox = bbox_list[i+2]
                x = int(face_bbox.split()[1])
                y = int(face_bbox.split()[2])
                w = int(face_bbox.split()[3])
                h = int(face_bbox.split()[4])
                if w < 48:
                    print('crop size is smaller than min size 48')
                    continue
                img = img[y:y+h,x:x+w]
                if img.size ==0:
                    continue
                print(img.size)
                #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img,(48,48))
                smile_label = int(attr_list[i+2].split()[32])
                gender_label = int(attr_list[i+2].split()[21])
                glasses_label = int(attr_list[i+2].split()[16])

                #get 50k images,25k smile and 25k not smile
                if (smile_label == 1) and (smile_count<20000):
                    X_smile.append((img,smile_label))
                    smile_count += 1
                if (smile_label == -1) and (notsmile_count<20000):
                    X_smile.append((img,smile_label))
                    notsmile_count += 1
                #get 40k images,20k male and 20k female
                if (gender_label == 1) and (male_count < 14000):
                    X_gender.append((img,gender_label))
                    male_count += 1
                if (gender_label == -1) and (female_count < 14000):
                    X_gender.append((img,gender_label))
                    female_count += 1
                #get all the glasses img(13k) and 13k not glasse img
                if (glasses_label == 1) and (glass_count < 13193):
                    X_glasses.append((img,glasses_label))
                    glass_count += 1
                if (glasses_label == -1) and (notglass_count<13193):
                    X_glasses.append((img,glasses_label))
                    notglass_count += 1
                print("{} {}:{} {}:{} {}:{}".format(img_list[i],attr_list[1].split()[31],attr_list[i+2].split()[32],attr_list[1].split()[15],attr_list[i+2].split()[16]
                                                    ,attr_list[1].split()[20],attr_list[i+2].split()[21]))
            for _ in range(10):
                np.random.shuffle(X_smile)
                np.random.shuffle(X_gender)
                np.random.shuffle(X_glasses)
            print('smile data size:%d'%len(X_smile))
            print('glasses data size:%d'%len(X_glasses))
            print('gender data size:%d'%len(X_gender))

            smile_boundary=int( (len(X_smile))*0.9 )
            gender_boundary=int( (len(X_gender))*0.9 )
            glasses_boundary=int( (len(X_glasses))*0.9 )

            train_data_smile, test_data_smile = X_smile[:smile_boundary], X_smile[smile_boundary:]
            train_data_gender, test_data_gender = X_gender[:gender_boundary], X_gender[gender_boundary:]
            train_data_glasses, test_data_glasses = X_glasses[:glasses_boundary],X_glasses[glasses_boundary:]

            np.save('./data/CelebA_color/' + 'train_smile.npy', train_data_smile)
            np.save('./data/CelebA_color/' + 'test_smile.npy', test_data_smile)

            np.save('./data/CelebA_color/' + 'train_glasses.npy', train_data_glasses)
            np.save('./data/CelebA_color/' + 'test_glasses.npy', test_data_glasses)

            np.save('./data/CelebA_color/' + 'train_gender.npy', train_data_gender)
            np.save('./data/CelebA_color/' + 'test_gender.npy', test_data_gender)

if __name__ == '__main__':
    get_crop_img(args['bbox'],args['image'],args['attr'])



