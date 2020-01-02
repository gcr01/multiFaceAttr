import os
import cv2
import pickle
import numpy as np
import argparse
import re

from mtcnn.mtcnn import MTCNN

ap=argparse.ArgumentParser()
ap.add_argument('--image',required=True,help='Path of Image')
args = vars(ap.parse_args())

def getBbox(image_path):
    detector=MTCNN()
    img_list=os.listdir(image_path)
    bboxes=[]
    nores=0
    #generate bbox info
    for name in img_list:
        img=cv2.imread(os.path.join(image_path,name))
        if img is None:
            print('loading img failed')
            continue
        height=img.shape[0]
        width=img.shape[1]
        result=detector.detect_faces(img)
        if not result:
            nores+=1
            print('detecting faces failed,'+'No.'+str(nores))
            continue
        face_position=result[0].get('box')
        if face_position[0]<0:
            face_position[0]=0
        if face_position[1]<0:
            face_position[1]=0
        if face_position[2]>width:
            face_position[2]=width
        if face_position[3]>height:
            face_position[3]=height
        bboxes.append((face_position,name))
    
    #save bbox info using open()
    with open('bboxes.txt','a') as txtfile:
        for bbox in bboxes:
            bboxinfo=bbox[0]
            img_name=bbox[-1]
            for info in bboxinfo:
                txtfile.write(str(info))
                txtfile.write(' ')
            txtfile.write(img_name)
            txtfile.write('\n')
    print('done')

if __name__ == '__main__' :
    getBbox(args['image'])

    

    



        


