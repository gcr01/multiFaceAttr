import os
import cv2
import pickle
import numpy as np
import argparse
import re
import csv
ap=argparse.ArgumentParser()
ap.add_argument('--image',required=True,
                help='Path of images')
ap.add_argument('--bboxes',required=True,
help = 'Path of bboxes info')
args = vars(ap.parse_args())


#1-116岁，0-115 array
def get_checked_age(img_path,bboxes_path):
    label_list = os.listdir(img_path)
    X_age = []
    age_list = [0 for i in range(116)]

    positive_list=[]  # etc. positive_list[0] images that are useful in dir20
    bboxes_list=[]    #  bboxes for useful images
    for i in range(51):
        positive_list.append([])
        bboxes_list.append([])

    fo=open(os.path.join(bboxes_path,'bboxes.txt'),'r')
    for line in fo.readlines():
        bboxinfo=line.strip().split()  #get bbox info with bbox cor and image name [x,y,w,h,name,dir]
        positive_list[int(bboxinfo[-1])-20].append(bboxinfo[-2])  #get useful img names
        bboxes_list[int(bboxinfo[-1])-20].append([int(i) for i in bboxinfo[:-2]]) #get bboxes of useful images
    fo.close()

    for label in label_list:
        dir_path=os.path.join(img_path,label)
        img_list=positive_list[int(label)-20]
        bbox_list=bboxes_list[int(label)-20]
        for name,bbox in zip(img_list,bbox_list):
            age = int(label)
            print('{} Real Age:{}'.format(name,str(age)))
            # age_list[age-1] += 1
            img = cv2.imread(os.path.join(dir_path,name))
            if img is None:
                continue
            face_position = bbox
            x_coordinate = face_position[0]
            y_coordinate = face_position[1]
            w_coordinate = face_position[2]
            h_coordinate = face_position[3]
            img = img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
            if (img.size == 0):
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))

            #========对年龄样本进行过采样=========
            if(age<1 or age>116):
                print('Wrong Age')
                continue
            
            elif(age>=20 and age <40):
                X_age.append((img, age))
                age_list[age-1] += 1
            elif(age>=40 and age <60):
                X_age.append((img, age))
                X_age.append((img, age))
                age_list[age-1]+=2
            elif(age>=60 and age<=70):
                X_age.append((img, age))
                X_age.append((img, age))
                X_age.append((img, age))
                age_list[age-1]+=3
            
    for _ in range(10):
        np.random.shuffle(X_age)
    print('data size : %d' % (len(X_age)))
    with open('checkedAge-oversampling-list.csv','a') as csvfile:
        writer = csv.writer(csvfile,delimiter = ',')
        for i in range(1,117):
            row = []
            row.append(str(i))
            row.append(str(age_list[i-1]))
            writer.writerow(row)
    age_boundary=int( (len(X_age))*0.9 )
    train_data_age, test_data_age = X_age[:age_boundary], X_age[age_boundary:]
    np.save('./data/imdb_checked/' + 'train_age_oversampling.npy', train_data_age)
    np.save('./data/imdb_checked/' + 'data_age_oversampling.npy', X_age)
    np.save('./data/imdb_checked/' + 'test_age_oversampling.npy', test_data_age)


if __name__ == '__main__':
    get_checked_age(args['image'],args['bboxes'])



