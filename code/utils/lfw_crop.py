import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import os
import re
import pickle
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--image",required=True,
               help='Path of the human face image')
args=vars(ap.parse_args())

def main(image_path):
    fail_detect = 0
    wrong_detect = 0
    detector = MTCNN()
    name_list = os.listdir(image_path)
    for name in name_list:
        name_path = os.path.join(image_path,name)
        img_list = os.listdir(name_path)
        crop_path = '/home/gucongrong/lfw_crop/' + name + '/' 
        #os.mkdir(crop_path)
        #os.makedirs(crop_path)
        for img in img_list:
            img_path = os.path.join(name_path,img)
            original_img = cv2.imread(img_path)
            result = detector.detect_faces(original_img)
            if not result:
                print('can not detect face in the photo')
                print(name_path + '/' + img)
                fail_detect += 1
                continue
            face_position = result[0].get('box')
            x_coordinate = face_position[0]
            y_coordinate = face_position[1]
            w_coordinate = face_position[2]
            h_coordinate = face_position[3]
            crop_img = original_img[y_coordinate: y_coordinate + h_coordinate, x_coordinate: x_coordinate + w_coordinate]
            if crop_img.size ==0:
                print('face img size is zero.Something wrong!')
                print(name_path + '/' + img)
                wrong_detect += 1
                continue
            crop_img = cv2.resize(crop_img, (112, 112))
            #cv2.imwrite(crop_path + img, crop_img)
    print('Detect Failed Times: ' + str(fail_detect))
    print('Wrong Detect Result Times: ' + str(wrong_detect))
if __name__ == '__main__':
    main(args['image'])
    










