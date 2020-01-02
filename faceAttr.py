import os
import numpy

#acc = (true predicted)/(all faces)
#recall = (true positive)(all positive faces)
path = '/home/crgu/Desktop/'

ltxt_list = os.listdir(path + '/label') #得到标签文件列表
ltxt_list.sort()
label_list = []  #[[],[],...]
for txt in ltxt_list:  #期望得到本测试集中所有人脸的真值标签,将他们存在格式为'[[face0],[face1],...]'的列表中,face0举例为'Y','F','Y','Y','N'
    with open(path + '/label/' + txt, 'r') as f:
        a = f.readlines()
        a = [line.split() for line in a]
        itertimes = int(a[0][2])
        for i in range(itertimes):
            face_list = []
            for j in range(4+i*8,9+i*8):
                face_list.append(a[j][2])
            label_list.append(face_list)

ptxt_list = os.listdir(path + '/predict') #得到预测文件列表
predict_list.sort()
predict_list = []  #[[],[],...]
for txt in ptxt_list:  #期望得到本测试集中所有人脸的预测值,将他们存在格式为'[[face0],[face1],...]'的列表中,face0举例为'Y','F','Y','Y','N'
                       #age,gender,glass,ethnic,smile 01234
    with open(path + '/predict/' + txt, 'r') as f:
        a = f.readlines()
        a = [line.split() for line in a]
        itertimes = int(a[0][2])
        for i in range(itertimes):
            face_list = []
            for j in range(4+i*8,9+i*8):
                face_list.append(a[j][2])
            predict_list.append(face_list)

def smile_acc_recall(label_list, predict_list):
    tpre = 0
    tpos = 0
    pos = 0
    for i in range(label_list):
        if label_list[i][4] == 'Y':
            pos += 1
            if (label_list[i][4] == 'Y') and (predict_list[i][4] == 'Y'):
                tpos += 1
        if label_list[i][4] == predict_list[i][4]:
            tpre += 1
        
    acc = tpre/len(label_list)
    recall = tpos/pos
    return acc,recall

def gender_acc_recall(label_list, predict_list):
    tpre = 0
    tpos = 0
    pos = 0
    for i in range(label_list):
        if label_list[i][1] == 'M':
            pos += 1
            if (label_list[i][1] == 'M') and (predict_list[i][1] == 'M'):
                tpos += 1
        if label_list[i][1] == predict_list[i][1]:
            tpre += 1
        
    acc = tpre/len(label_list)
    recall = tpos/pos
    return acc,recall
def glass_acc_recall(label_list, predict_list):
def ethnic_acc(label_list, predict_list):
def age_acc(label_list, predict_list):

def white_recall():
def black_recall():
def yellow_recall():

def child_recall():
def teen_recall():
def young_recall():
def middle_recall():
def old_recall():









    

    








def get_recall():

if '__name__' == '__main__':
    smile_acc