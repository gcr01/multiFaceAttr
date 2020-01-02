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
def get_checked_age(img_path):
    detector = MTCNN()
    label_list = os.listdir(img_path)
    X_age = []
    out_imgs = []
    out_ages = []
    age_list = [0 for i in range(116)]
    for label in label_list:
        final_path=os.path.join(img_path,label)
        img_list=os.listdir(final_path)
        for name in img_list:
            real_age = int(label)
            print('{} Real Age:{}'.format(name,str(real_age)))
            if(real_age<1 or real_age>116):
                print('age out of range')
                continue
            age_list[real_age-1] += 1
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
            img = img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
            if (img.size == 0):
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size, img_size))
            X_age.append((img,real_age))
            # out_imgs.append(img)
            # out_ages.append(real_age)
    for _ in range(10):
        np.random.shuffle(X_age)
    print('data size : %d' % (len(X_age)))
    # with open('checkedAge-list.csv','a') as csvfile:
    #     writer = csv.writer(csvfile,delimiter = ',')
    #     for i in range(1,117):
    #         row = []
    #         row.append(str(i))
    #         row.append(str(age_list[i-1]))
    #         writer.writerow(row)
    # age_boundary=int( (len(out_imgs))*0.9 )
    age_boundary=int( (len(X_age))*0.9 )
    # train_path = './imdb_train'
    # test_path = './imdb_test'
    # train_out_imgs, test_out_imgs = out_imgs[:age_boundary], out_imgs[age_boundary:]
    # train_out_ages, test_out_ages = out_ages[:age_boundary], out_ages[age_boundary:]
    #np.save('./data/imdb_checked/' + 'train_age_oversampling.npy', train_data_age)
    #np.save('./data/imdb_checked/' + 'data_age_oversampling.npy', X_age)
    np.save('./data/testAge/' + 'test_age.npy', X_age)
    # np.savez(train_path,image=np.array(train_out_imgs),age=np.array(train_out_ages),img_size=img_size)
    # np.savez(test_path,image=np.array(test_out_imgs),age=np.array(test_out_ages),img_size=img_size)

if __name__ == '__main__':
    get_checked_age(args['image'])



