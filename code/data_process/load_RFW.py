import os
import numpy as np
import re
import cv2
import argparse
import csv
from mtcnn.mtcnn import MTCNN

ap = argparse.ArgumentParser()
ap.add_argument('--image',required=True,
                help = 'Path of Image')
                # ../human_database/RFW/test/data/
args = vars(ap.parse_args())
race_dict={0:'White',1:'Black',2:'Asian',3:'Indian',4:'Other'}

# RFW:race
def get_rfw(image_path):

    label_list=os.listdir(image_path) # images that are useful
    X_race =[]
    # race_list = [0,0,0,0,0]
    i=0
    small = 0
    #generating positive_list and bboxes
    detector=MTCNN(min_face_size = 48) #过滤掉一些比较小的脸图，防止在resize时候产生信息的无中生有
    for label in label_list:
        if label=='Caucasian':
            race=0
        elif label=='African':
            race=1
        elif label=='Asian':
            race=2
        elif label=='Indian':
            race=3
        else:
            race=4
        indetity_list=os.listdir(image_path+label)
        # if name=='.jpg.chip.jpg':
        #     print('wrong name')
        #     continue
        for indentity in indetity_list:
            final_path=os.path.join(image_path+label,indentity)
            img_list=os.listdir(final_path)
            for name in img_list:
                img = cv2.imread(os.path.join(final_path,name))
                if img is None:
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
                if w_coordinate < 48:
                    print('crop size is smaller than min size 48')
                    small += 1
                    continue
                img = img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                if (img.size == 0):
                    continue
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                print('RACE:{}'.format(race_dict[race]))
                i+=1
                print(i)
                # ========对人种样本进行采样=========
                if (race < 0 or race > 4):
                    print('Wrong Race')
                    continue
                X_race.append((img,race))
                # race_list[race]+=1
    print(small)
    for _ in range(10):
        np.random.shuffle(X_race)
    print('race data size : %d' % (len(X_race)))
    race_boundary=int( (len(X_race))*0.9 )
    train_data_race, test_data_race = X_race[:race_boundary], X_race[race_boundary:]
    np.save('./data/RFW_color/' + 'train_race.npy', train_data_race)
    np.save('./data/RFW_color/' + 'test_race.npy', test_data_race)
if __name__ == '__main__':
    get_rfw(args['image'])
