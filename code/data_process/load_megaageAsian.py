import os
import cv2
import pickle
import numpy as np
import argparse
import csv
from mtcnn.mtcnn import MTCNN

ap=argparse.ArgumentParser()
ap.add_argument('--image',required=True,help='Path of images')
args=vars(ap.parse_args())

img_size=48
'''
0-11 :chilid, 0
12-17:teen, 1
18-44:the young, 2 
45-65:the middle-aged, 3
65 ~ : the old 4
'''
age_dict = {0:'child', 1:'teen', 2:'the young', 3:'the middle-aged', 4:'the old'}
def get_asian_age(img_path):
    small = 0
    X_age = []
    age_list = [0 for i in range(5)]
    detector=MTCNN(min_face_size = 48) #过滤掉一些比较小的脸图，防止在resize时候产生信息的无中生有
    with open(img_path+'list/test_name.txt','r') as name:
        img_list=[line.strip() for line in name.readlines()]
    with open(img_path+'list/test_age.txt','r') as label:
        label_list=[line.strip() for line in label.readlines()]
    young_count = 0
    for i in range(len(img_list)):
        age=int(label_list[i])
        #####age classfication 
        if age >= 0 and age < 12:
            age = 0  #use age group to cover age
            age_list[age] += 1
        elif age >= 12 and age < 18:
            age = 1
            age_list[age] += 1
        elif age >= 12 and age < 45:
            age = 2
            age_list[age] += 1
        elif age >= 45 and age < 65:
            age = 3
            age_list[age] += 1
        else:
            age = 4
            age_list[age] += 1

        img_name=os.path.join(img_path+'test/',img_list[i])
        img=cv2.imread(img_name)
        if img is None:
            continue
        result=detector.detect_faces(img)
        if not result:
            continue
        face_pos=result[0].get('box')
        x=face_pos[0]
        y=face_pos[1]
        w=face_pos[2]
        h=face_pos[3]
        if w < 48:
            print('crop size is smaller than min size 48')
            small += 1
            continue
        img=img[y:y+h,x:x+w]
        if (img.size == 0):
            continue
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(img_size,img_size))
        print(img.shape)
        if age != 2 :
            X_age.append((img,age))
            print('{} Age:{}'.format(img_list[i],age_dict[age]))
        elif young_count < 15000 :
            X_age.append((img,age))
            print('{} Age:{}'.format(img_list[i],age_dict[age]))
            young_count += 1
    print(small)
    for _ in range(10):
        np.random.shuffle(X_age)
    print('data size : %d' % (len(X_age)))
    with open('mega-list.csv','a') as csvfile:
        writer = csv.writer(csvfile,delimiter = ',')
        for i in range(5):
            row = []
            row.append(str(i))
            row.append(str(age_list[i]))
            writer.writerow(row)
    # np.save('./data/megaage_asian_tripple/'+'train_age.npy',train_data_age)
    # np.save('./data/megaage_asian_tripple/'+'test_age.npy',test_data_age)
    np.save('./data/megaage_asian_color/'+'test_age.npy',X_age)

if __name__ == "__main__":
    get_asian_age(args['image'])
    
