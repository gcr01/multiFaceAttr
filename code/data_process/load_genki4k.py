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
#-1:not smile，1:smile
def get_smile_data(img_path):
    detector = MTCNN()
    label_list = os.listdir(img_path)
    X_smile = []
    for label in label_list:
        final_path=os.path.join(img_path,label)
        img_list=os.listdir(final_path)
        for name in img_list:
            label = int(label)
            print('{} emotion:{}'.format(name,str(label)))
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
            X_smile.append((img,label))
    for _ in range(10):
        np.random.shuffle(X_smile)

    train_data_imgs, test_data_imgs = X_smile,X_smile
    np.save('./data/imdb_checked/' + 'train_smile.npy', train_data_imgs)
    np.save('./data/imdb_checked/' + 'test_smile.npy', test_data_imgs)

if __name__ == '__main__':
    get_smile_data(args['image'])



