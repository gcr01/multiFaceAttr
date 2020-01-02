import os
import numpy as np
import re
import cv2
import argparse
import csv
from mtcnn.mtcnn import MTCNN

# ap = argparse.ArgumentParser()
# ap.add_argument('--imageOfRaf',required=True,
#                 help = 'Path of Image')
# ap.add_argument('--labelOfRaf',required=True,
#                 help = 'Path of label')
# ap.add_argument('--imageOfScut',required=True,
#                 help = 'Path of Image')
# args = vars(ap.parse_args())
rafPath='E:\gucongrong\Dataset\RAF-DB\compound\Image\original\original'
rafLabel='E:\gucongrong\Dataset\RAF-DB\compound\Annotation\manual\manual'
scutPath='E:\gucongrong\Dataset\SCUT-FBP5500_v2\SCUT-FBP5500_v2\Images'
scut_dict={'A':2,'C':0}
race_dict={0:'White',1:'Black',2:'Asian',3:'Indian',4:'Other'}

detector = MTCNN()

def get_raf(image_path,label_path):

    img_list = os.listdir(image_path)
    label_list=os.listdir(label_path)
    X_race =[]
    race_list = [0,0,0,0,0]
    for name,label in zip(img_list,label_list):
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
        img = img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
        if (img.size == 0):
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))

        fo = open(os.path.join(label_path,label), 'r')
        race_=[]
        for line in fo.readlines():
            # fo.readlines()读取整个文件，返回数据中每一行有\n存在，需使用line.strip去掉
            # readlines()方法读取整个文件所有行，保存在一个列表(list)变量中，每行作为一个元素，但读取大文件会比较占内存
            line = line.strip()   # 去掉每行头尾空白
            race_.append(line)
        race=int(race_[6])
        
        print('{} RACE:{}'.format(name,race_dict[race]))
       # ========对人种样本进行采样=========
        if (race < 0 or race > 4):
            print('Wrong Race')
            continue
        X_race.append((img,race))
        race_list[race]+=1

    with open('raf_race_list.csv','a') as csvfile:  #对人种样本分布进行统计
        writer = csv.writer(csvfile,delimiter = ',')
        for i in range(5):
            temp = []
            temp.append(race_dict[i])
            temp.append(str(race_list[i]))
            writer.writerow((race_dict[i],str(race_list[i])))
    print('race data size : %d' % (len(X_race)))
    return X_race,race_list

def get_scut(image_path):
    img_list=os.listdir(image_path)
    X_race =[]
    race_list = [0,0,0,0,0]
    for name in img_list:
        race = scut_dict[name[0]]
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
        img = img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
        if (img.size == 0):
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        print('{} RACE:{}'.format(name,race_dict[race]))
        # ========对人种样本进行采样=========
        if (race < 0 or race > 4):
            print('Wrong Race')
            continue
        X_race.append((img,race))
        race_list[race]+=1
    with open('scut_race_list.csv','a') as csvfile:  #对人种样本分布进行统计
        writer = csv.writer(csvfile,delimiter = ',')
        for i in range(5):
            temp = []
            temp.append(race_dict[i])
            temp.append(str(race_list[i]))
            writer.writerow((race_dict[i],str(race_list[i])))
    # for _ in range(10):
    #     np.random.shuffle(X_race)
    print('race data size : %d' % (len(X_race)))
    return X_race,race_list

def get_all():
    X_race_r,race_list_r=get_raf(rafPath,rafLabel)
    X_race_s,race_list_s=get_scut(scutPath)
    X_race=X_race_r+X_race_s
    race_list = [i+j for i, j in zip(race_list_r, race_list_s)]
    with open('raf_scut_race_list.csv','a') as csvfile:  #对人种样本分布进行统计
        writer = csv.writer(csvfile,delimiter = ',')
        for i in range(5):
            temp = []
            temp.append(race_dict[i])
            temp.append(str(race_list[i]))
            writer.writerow((race_dict[i],str(race_list[i])))
    for _ in range(10):
        np.random.shuffle(X_race)
    train_data_race, test_data_race = X_race[:6930], X_race[6930:]
    print('race data size : %d' % (len(X_race)))
    np.save('./data/raf_scut/' + 'train_race.npy', train_data_race)
    np.save('./data/raf_scut/' + 'data_race.npy', X_race)
    np.save('./data/raf_scut/' + 'test_race.npy', test_data_race)
    
    
if __name__ == '__main__':
    get_all()
    