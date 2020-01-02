import os
import numpy as np
import re
import cv2
import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument('--image',required=True,
                help = 'Path of Image')
args = vars(ap.parse_args())
race_dict={0:'White',1:'Black',2:'Asian',3:'Indian',4:'Other'}
gender_dict={0:'male',1:'female'}
def get_utkface(image_path):
    img_list = os.listdir(image_path)
    X_age =[]
    X_race =[]
    X_gender=[]
    race_list = [0,0,0,0,0]
    age_list = [0 for i in range(117)]
    gender_list=[0,0]
    for name in img_list:
        age = int(re.findall('\d+',name)[0])
        gender = int(re.findall('\d+',name)[1])
        race = int(re.findall('\d+',name)[2])
        img = cv2.imread(os.path.join(image_path,name))
        if not img.any():
            print('can not open the image')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        print('{} AGE:{} GENDER:{} RACE:{}'.format(name,str(age),gender_dict[gender],race_dict[race]))
        if(age<0 or age>116):
            print('Wrong Age')
            continue


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

        
    #     #========对年龄样本进行过采样=========
    #     if(age>=0 and age<10):
    #         X_age.append((img,age))
    #         X_age.append((img, age))
    #         age_list[age] += 2
    #     elif((age>=10 and age <20) or (age>=60 and age <116)):
    #         X_age.append((img, age))
    #         X_age.append((img, age))
    #         X_age.append((img, age))
    #         X_age.append((img, age))
    #         age_list[age] += 4
    #     elif(age>=20 and age <30):
    #         X_age.append((img, age))
    #         age_list[age] += 1
    #     elif(age>=30 and age <40):
    #         X_age.append((img, age))
    #         X_age.append((img, age))
    #         age_list[age]+=2
    #     elif(age>=40 and age<60):
    #         X_age.append((img, age))
    #         X_age.append((img, age))
    #         X_age.append((img, age))
    #         age_list[age]+=3

    #    # ========对人种样本进行过采样=========
    #     if (race < 0 or race > 4):
    #         print('Wrong Race')
    #         continue
    #     if(race == 0):
    #         X_race.append((img,race))
    #         race_list[race]+=1

    #     if(race == 1):  #balance the number of the class
    #         X_race.append((img,race))
    #         X_race.append((img,race))
    #         race_list[race] += 2

    #     elif(race == 2 or race==3):
    #         X_race.append((img,race))
    #         X_race.append((img,race))
    #         X_race.append((img,race))
    #         race_list[race]+=3

    #     elif(race==4):
    #         for i in range(6):
    #             X_race.append((img,race))
    #             race_list[race]+=1
        
    #     #####对性别进行采样#####
    #     if (gender!=0 and gender!=1):
    #         print('Wrong gender')
    #         continue
    #     X_gender.append((img,gender))
    #     gender_list[gender] += 1

    # with open('utk_race_list.csv','a') as csvfile:  #对过采样后的人种样本分布进行统计
    #     writer = csv.writer(csvfile,delimiter = ',')
    #     for i in range(5):
    #         temp = []
    #         temp.append(race_dict[i])
    #         temp.append(str(race_list[i]))
    #         writer.writerow((race_dict[i],str(race_list[i])))
    # with open('utk_age_list.csv','a') as csvfile:  #对过采样后的年龄样本分布进行统计
    #     writer = csv.writer(csvfile,delimiter = ',')
    #     for i in range(117):
    #         temp = []
    #         temp.append(str(i))
    #         temp.append(str(age_list[i]))
    #         writer.writerow(temp)
    for _ in range(10):
        np.random.shuffle(X_age)
        np.random.shuffle(X_race)
        np.random.shuffle(X_gender)
    print('age data size : %d' % (len(X_age)))
    print('gender data size : %d' % (len(X_gender)))
    print('race data size : %d' % (len(X_race)))

    race_boundary=int( (len(X_race))*0.9 )
    age_boundary=int( (len(X_age))*0.9 )
    gender_boundary=int( (len(X_gender))*0.9 )
    
    train_data_age, test_data_age = X_age[:age_boundary], X_age[age_boundary:]
    np.save('./data/utkface/' + 'train_age_2070.npy', train_data_age)
    np.save('./data/utkface/' + 'data_age_2070.npy', X_age)
    np.save('./data/utkface/' + 'test_age_2070.npy', test_data_age)
    
    # train_data_race, test_data_race = X_race[:race_boundary], X_race[race_boundary:]
    # np.save('./data/utkface/' + 'train_race.npy', train_data_race)
    # np.save('./data/utkface/' + 'data_race.npy', X_race)
    # np.save('./data/utkface/' + 'test_race.npy', test_data_race)
    
    # train_data_gender, test_data_gender = X_gender[:gender_boundary], X_gender[gender_boundary:]
    # np.save('./data/utkface/' + 'train_gender.npy', train_data_gender)
    # np.save('./data/utkface/' + 'data_gender.npy', X_gender)
    # np.save('./data/utkface/' + 'test_gender.npy', test_data_gender)
if __name__ == '__main__':
    get_utkface(args['image'])