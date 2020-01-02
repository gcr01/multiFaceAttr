import os
import numpy as np
import re
import pandas as pd
import cv2
import argparse
import csv
from mtcnn.mtcnn import MTCNN

#for beauty scoring
def get_scut(image_path):
    detector=MTCNN(min_face_size = 48) #过滤掉一些比较小的脸图，防止在resize时候产生信息的无中生有
    rating = np.asarray(pd.read_excel('E:/gucongrong/Dataset/SCUT-FBP5500_v2/mean_ratings.xlsx'))
    X_beauty =[]
    beauty_list = [0,0,0,0,0]
    for i in range(len(rating)): 
        name = rating[i][0]
        label = round(rating[i][1])
        img = cv2.imread(os.path.join(image_path,name))
        if not img.any():
            print('can not open the image')
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
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        print('{} Score:{}'.format(name,label))
        # ========对颜值样本进行采样=========
        X_beauty.append((img,label))
        beauty_list[label - 1] += 1
    with open('beauty_list.csv','a') as csvfile:  #对颜值样本分布进行统计
        writer = csv.writer(csvfile,delimiter = ',')
        for i in range(5):
            writer.writerow((str(beauty_list[i])))
    for _ in range(10):
        np.random.shuffle(X_beauty)
    print('beauty data size : %d' % (len(X_beauty)))
    beauty_boundary = int(0.9 * len(X_beauty))
    train_data_beauty, test_data_beauty = X_beauty[:beauty_boundary], X_beauty[beauty_boundary:]
    train_data_beauty = np.concatenate((train_data_beauty,train_data_beauty,train_data_beauty), axis = 0)
    for _ in range(15):
        np.random.shuffle(train_data_beauty)
    print('Training Data After 3X OverSampling : %d' % (len(train_data_beauty)) ) 

    np.save('E:/gucongrong/FaceAttributeDetection/data/Scut/' + 'train_beauty.npy', train_data_beauty)
    np.save('E:/gucongrong/FaceAttributeDetection/data/Scut/'+ 'test_beauty.npy', test_data_beauty)

if __name__ == '__main__':
    image_path = 'E:/gucongrong/Dataset/SCUT-FBP5500_v2/Images/'
    get_scut(image_path)

