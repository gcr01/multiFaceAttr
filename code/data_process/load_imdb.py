import os
import cv2
import pickle
import numpy as np
import argparse
import re
import scipy.io as sio
import datetime
from mtcnn.mtcnn import MTCNN
import csv
ap=argparse.ArgumentParser()
ap.add_argument('--image',required=True,
                help='Path of images')
ap.add_argument('--mat',required=True,
                help='Path of mat file')
args = vars(ap.parse_args())


def reformat_date(mat_date):
    dt = datetime.date.fromordinal(np.max([mat_date - 366, 1])).year
    return dt
def get_crop_img(img_path,mat_path):
    mat_struct = sio.loadmat(mat_path)
    data_set = [data[0] for data in mat_struct['imdb'][0, 0]]

    keys = ['dob',
            'photo_taken',
            'full_path',
            'gender',
            'name',
            'face_location',
            'face_score',
            'second_face_score',
            'celeb_names',
            'celeb_id'
            ]
    print("Dictionary created...")
    imdb_dict = dict(zip(keys, np.asarray(data_set)))
    #raw_face_location = imdb_dict['face_location']
    X_age=[]
    imdb_dict['dob'] = [reformat_date(dob) for dob in imdb_dict['dob']]
    imdb_dict['age'] = imdb_dict['photo_taken'] - imdb_dict['dob']
    imgs = imdb_dict['full_path']
    i=0
    for name in imgs:
        img = cv2.imread(os.path.join(img_path,name[0]))
        print(os.path.join(img_path,name[0]))
        # birth_y = int(re.findall('\d+',name[0])[3])
        # photo_y = int(re.findall('\d+',name[0])[6])
        real_age = int(imdb_dict['age'][i])
        if len(img)==0:
            continue
        print(img.size)
        # face_box = raw_face_location[i][0]
        # img = img[face_box[1]:face_box[3],face_box[0]:face_box[4]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        print("{} Birth Year:{} Photo Year:{} Real Age:{}".format(name, str(imdb_dict['dob'][i]),
                                                                  str(imdb_dict['photo_taken'][i]), str(real_age)))
        i += 1
        if(real_age<1 or real_age>101):
            print('================='+name[0]+'age is '+str(real_age)+'=================')
            continue
        X_age.append((img,real_age))
    for _ in range(10):
        np.random.shuffle(X_age)
    print('data size : %d'%(len(X_age)))
    train_data_age , test_data_age = X_age[:360000],X_age[360000:]
    np.save('./data/imdb/'+'train_age.npy',train_data_age)
    np.save('./data/imdb/' + 'data_age.npy', X_age)
    np.save('./data/imdb/' + 'test_age.npy', test_data_age)
def get_checked_imdb(img_path):
    img_list = os.listdir(img_path)
    #detector = MTCNN()
    X_age = []
    age_list = [0 for i in range(1,102)]
    for name in img_list:
        birth_year = re.findall('\d+', name)[2]
        photo_year = re.findall('\d+',name)[5]
        real_age = int(photo_year)-int(birth_year)
        print('{} Birth Year:{} Photo Year:{} Real Age:{}'.format(name, birth_year, photo_year, str(real_age)))
        if(real_age<1 or real_age>101):
            print('age out of range')
            continue
        age_list[real_age] += 1
        img = cv2.imread(os.path.join(img_path,name))
        if not img.all():
            continue
        # result = detector.detect_faces(img)
        # if not result:
        #     continue
        # face_position = result[0].get('box')
        # x_coordinate = face_position[0]
        # y_coordinate = face_position[1]
        # w_coordinate = face_position[2]
        # h_coordinate = face_position[3]
        # img = img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
        # if (img.size == 0):
        #     continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        X_age.append((img,real_age))
    for _ in range(10):
        np.random.shuffle(X_age)
    print('data size : %d' % (len(X_age)))
    with open('age-list.csv','a') as csvfile:
        writer = csv.writer(csvfile,delimiter = ',')
        for i in range(101):
            row = []
            row.append(str(i))
            row.append(str(age_list[i]))
            writer.writerow(row)
    train_data_age, test_data_age = X_age[:100000], X_age[100000:]
    np.save('./data/imdb/' + 'train_age.npy', train_data_age)
    np.save('./data/imdb/' + 'data_age.npy', X_age)
    np.save('./data/imdb/' + 'test_age.npy', test_data_age)
if __name__ == '__main__':
    get_checked_imdb(args['image'])



