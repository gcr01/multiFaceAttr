import os
import cv2
import pickle
import numpy as np
import argparse
import re

ap=argparse.ArgumentParser()
ap.add_argument('--bboxes',required=True,help='Path of bboxes')
ap.add_argument('--images',required=True,help='Path of images')
args = vars(ap.parse_args())

def bboxesScaling(images_path,bboxes_path):
    bboxinfo=[]  
    bboxes_list=[]
    fo=open(os.path.join(bboxes_path,'bboxes.txt'),'r')
    for line in fo.readlines():
        bboxinfo=line.strip().split()  #get bbox info with bbox cor and image name
        img_name=bboxinfo[-2]
        label_name=bboxinfo[-1]

        #get the width&height of current image
        img=cv2.imread(os.path.join(images_path,img_name))
        height=img.shape[0]
        width=img.shape[1]
        bboxes_list.append([int(i) for i in bboxinfo[:-2]])   #for imdb_checked store by age directory
        bbox=bboxes_list[-1]
        x=bbox[0]
        y=bbox[1]
        w=bbox[2]
        h=bbox[3]
        #边框原点(x,y)向左上移动
        bbox[0]-=(0.25*w)
        bbox[0] = int(bbox[0])
        if bbox[0]<0:
            bbox[0]=0
        bbox[1]-=(0.25*h)
        bbox[1] = int(bbox[1])
        if bbox[1]<0:
            bbox[1]=0
        #边框宽高(w,h)增长
        bbox[2]*=1.5
        bbox[2] = int(bbox[2])
        if bbox[2]>width:
            bbox[2]=width
        bbox[3]*=1.5
        bbox[3] = int(bbox[3])
        if bbox[3]>height:
            bbox[3]=height
        bboxes_list[-1]=(bbox,img_name,label_name)
    fo.close()

    #save bbox info using open()
    with open('bboxes_scale.txt','a') as txtfile:
        for bbox in bboxes_list:
            bboxinfo=bbox[0]
            img_name=bbox[-2]
            label_name=bbox[-1]
            for info in bboxinfo:
                txtfile.write(str(info))
                txtfile.write(' ')
            txtfile.write(img_name)
            txtfile.write(' ')
            txtfile.write(label_name)
            txtfile.write('\n')
    print('done')

if __name__ == '__main__' :
    bboxesScaling(args['images'],args['bboxes'])