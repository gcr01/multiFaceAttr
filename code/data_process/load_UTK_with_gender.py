import os
import numpy as np
import re
import cv2
import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument('--image',required=True,
                help = 'Path of Image')
ap.add_argument('--bboxes',required=True,
                help = 'Path of bboxes info')
args = vars(ap.parse_args())
race_dict={0:'White',1:'Black',2:'Asian',3:'Indian',4:'Other'}
gender_dict={0:'male',1:'female'}

#utk :age/gender/race
def get_utkface(image_path,bboxes_path):
    positive_list=[] # images that are useful
    bboxes_list=[]  #bboxes for useful images
    X_age =[]
    X_gender=[]
    X_race =[]
    race_list = [0,0,0,0,0]
    age_list = [0 for i in range(117)]
    gender_list=[0,0]

    #generating positive_list and bboxes
    fo=open(os.path.join(bboxes_path,'bboxes.txt'),'r')
    for line in fo.readlines():
        bboxinfo=line.strip().split()  #get bbox info with bbox cor and image name
        positive_list.append(bboxinfo[-1])  #get useful img names
        bboxes_list.append([int(i) for i in bboxinfo[:-1]]) #get bboxes of useful images
    fo.close()

    for name,bbox in zip(positive_list,bboxes_list):
        if name=='.jpg.chip.jpg':
            print('wrong name')
            continue
        age = int(re.findall('\d+',name)[0])
        gender = int(re.findall('\d+',name)[1])
        race = int(re.findall('\d+',name)[2])
        img = cv2.imread(os.path.join(image_path,name))
        if img is None:
            print('can not open the image')
            continue
        x_cor=bbox[0]
        y_cor=bbox[1]
        w_cor=bbox[2]
        h_cor=bbox[3]
        img=img[y_cor:y_cor+h_cor,x_cor:x_cor+w_cor]
        if img.size==0:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        print('{} AGE:{} GENDER:{} RACE:{}'.format(name,str(age),gender_dict[gender],race_dict[race]))
        
        ######对20-70岁年龄进行采样做测试集用来测试IMDBCHECKED#####
        if(age<20 or age>70):
            print('Wrong Age')
            continue
        if(age>=20 and age<30):
            X_age.append((img,age))
            age_list[age] += 1
        elif(age>=30 and age<40):
            X_age.append((img,age))
            X_age.append((img,age))
            age_list[age] += 2
        elif(age>=40 and age<=70):
            X_age.append((img,age))
            X_age.append((img,age))
            age_list[age] += 2

        # #========对年龄样本进行过采样=========
        # if(age<0 or age>116):
        #     print('Wrong Age')
        #     continue
        # if(age>=0 and age<10):
        #     X_age.append((img,age))
        #     X_age.append((img, age))
        #     age_list[age] += 2
        # elif((age>=10 and age <20) or (age>=60 and age <116)):
        #     X_age.append((img, age))
        #     X_age.append((img, age))
        #     X_age.append((img, age))
        #     X_age.append((img, age))
        #     age_list[age] += 4
        # elif(age>=20 and age <30):
        #     X_age.append((img, age))
        #     age_list[age] += 1
        # elif(age>=30 and age <40):
        #     X_age.append((img, age))
        #     X_age.append((img, age))
        #     age_list[age]+=2
        # elif(age>=40 and age<60):
        #     X_age.append((img, age))
        #     X_age.append((img, age))
        #     X_age.append((img, age))
        #     age_list[age]+=3

        # #####对性别进行采样#####
        # if (gender!=0 and gender!=1):
        #     print('Wrong gender')
        #     continue
        # X_gender.append((img,gender))
        # gender_list[gender] += 1
        
        # # #####对种族进行采样#####
        # # if (race < 0 or race > 4):
        # #     print('Wrong race')
        # #     continue
        # # X_race.append((img,race))
        # # race_list[race] += 1

        # # ========对人种样本进行过采样=========
        # if (race < 0 or race > 4):
        #     print('Wrong Race')
        #     continue
        # if(race == 0):
        #     X_race.append((img,race))
        #     race_list[race]+=1

        # if(race == 1):  
        #     X_race.append((img,race))
        #     X_race.append((img,race))
        #     race_list[race] += 2

        # elif(race == 2 or race==3):
        #     X_race.append((img,race))
        #     X_race.append((img,race))
        #     X_race.append((img,race))
        #     race_list[race]+=3

        # elif(race==4):
        #     for i in range(6):
        #         X_race.append((img,race))
        #         race_list[race]+=1


    # with open('utk_race_oversampling_list.csv','a') as csvfile:  #对人种样本分布进行统计
    #     writer = csv.writer(csvfile,delimiter = ',')
    #     for i in range(5):
    #         temp = []
    #         temp.append(race_dict[i])
    #         temp.append(str(race_list[i]))
    #         writer.writerow((race_dict[i],str(race_list[i])))
    # with open('utk_age_oversampling_list.csv','a') as csvfile:  #对年龄样本分布进行统计
    #     writer = csv.writer(csvfile,delimiter = ',')
    #     for i in range(117):
    #         temp = []
    #         temp.append(str(i))
    #         temp.append(str(age_list[i]))
    #         writer.writerow(temp)
    # with open('utk_gender_list.csv.csv','a') as csvfile:  #对性别样本分布进行统计
    #     writer = csv.writer(csvfile,delimiter = ',')
    #     for i in range(2):
    #         temp = []
    #         temp.append(str(i))
    #         temp.append(str(gender_list[i]))
    #         writer.writerow((gender_dict[i],str(gender_list[i])))
    for _ in range(10):
        np.random.shuffle(X_age)
        np.random.shuffle(X_gender)
        np.random.shuffle(X_race)
    print('age data size : %d' % (len(X_age)))
    print('gender data size : %d' % (len(X_gender)))
    print('race data size : %d' % (len(X_race)))

    race_boundary=int( (len(X_race))*0.9 )
    age_boundary=int( (len(X_age))*0.9 )
    gender_boundary=int( (len(X_gender))*0.9 )
    
    train_data_age, test_data_age = X_age[:age_boundary], X_age[age_boundary:]
    np.save('./data/utkface_crop/' + 'train_age_2070.npy', train_data_age)
    np.save('./data/utkface_crop/' + 'data_age_2070.npy', X_age)
    np.save('./data/utkface_crop/' + 'test_age_2070.npy', test_data_age)
    
    # train_data_race, test_data_race = X_race[:race_boundary], X_race[race_boundary:]
    # np.save('./data/utkface_crop/' + 'train_race.npy', train_data_race)
    # np.save('./data/utkface_crop/' + 'data_race.npy', X_race)
    # np.save('./data/utkface_crop/' + 'test_race.npy', test_data_race)
    
    # train_data_gender, test_data_gender = X_gender[:gender_boundary], X_gender[gender_boundary:]
    # np.save('./data/utkface_crop/' + 'train_gender.npy', train_data_gender)
    # np.save('./data/utkface_crop/' + 'data_gender.npy', X_gender)
    # np.save('./data/utkface_crop/' + 'test_gender.npy', test_data_gender)
if __name__ == '__main__':
    get_utkface(args['image'],args['bboxes'])