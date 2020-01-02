import numpy as np
import os
import cv2
import scipy
import random

DATA_FOLDER = '/home/gucongrong/FaceAttributeDetection/data/'
#DATA_FOLDER = '/home/xiaoyang/gcr_FaceAttr/data/'
#DATA_FOLDER = 'E:/gucongrong/FaceAttributeDetection/data/'

NUM_SMILE_IMAGE = 4000
SMILE_SIZE = 48
EMOTION_SIZE = 48


def getSmileImage():
    print('Load smile image...................')
    X1 = np.load(DATA_FOLDER + 'CelebA_color/train_smile.npy',allow_pickle=True)
    X2 = np.load(DATA_FOLDER + 'CelebA_color/test_smile.npy',allow_pickle=True)

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of smile train data: ',str(len(train_data)))
    print('Number of smile test data: ',str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

def getGenderImage():
    print('Load gender image...................')
    X1 = np.load(DATA_FOLDER + 'CelebA_color/train_gender.npy',allow_pickle=True)
    X2 = np.load(DATA_FOLDER + 'CelebA_color/test_gender.npy',allow_pickle=True)

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')

    print('Number of gender train data: ', str(len(train_data)))
    print('Number of gender test data: ',str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

def getAfadGenderImage():
    print('Load afadGender image...................')
    X1 = np.load(DATA_FOLDER + 'afad-lite/train_gender.npy',allow_pickle=True)
    X2 = np.load(DATA_FOLDER + 'afad-lite/test_gender.npy',allow_pickle=True)

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')

    print('Number of afadgender train data: ', str(len(train_data)))
    print('Number of afadgender test data: ',str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

def getGlassesImage():
    print('Load Glasses image...................')
    X1 = np.load(DATA_FOLDER + 'CelebA_color/train_glasses.npy',allow_pickle=True)
    X2 = np.load(DATA_FOLDER + 'CelebA_color/test_glasses.npy',allow_pickle=True)

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of glasses train data: ', str(len(train_data)))
    print('Number of glasses test data: ',str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

def getEthnicImage():
    print('Load ethnic image...................')
    X1 = np.load(DATA_FOLDER + 'RFW_color/train_race.npy', allow_pickle=True)
    X2 = np.load(DATA_FOLDER + 'RFW_color/test_race.npy', allow_pickle=True)
    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of ethnic train data: ', str(len(train_data)))
    print('Number of ethnic test data: ',str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

def getAgeImage():
    print('Load age image...................')
    X1 = np.load(DATA_FOLDER + 'megaage_asian_color/train_age.npy',allow_pickle=True)
    X2 = np.load(DATA_FOLDER + 'megaage_asian_color/test_age.npy',allow_pickle=True)

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of age train data: ', str(len(train_data)))
    print('Number of age test data: ', str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

def getAfadAgeImage():
    print('Load afadage image...................')
    X1 = np.load(DATA_FOLDER + 'afad-full/train_age.npy',allow_pickle=True)
    X2 = np.load(DATA_FOLDER + 'afad-full/test_age.npy',allow_pickle=True)
    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])
    print('Done !')
    print('Number of age train data: ', str(len(train_data)))
    print('Number of age test data: ', str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data


def getBeautyImage():
    print('Load beauty image...................')
    X1 = np.load(DATA_FOLDER + 'beauty/train_age.npy',allow_pickle=True)
    X2 = np.load(DATA_FOLDER + 'beauty/test_age.npy',allow_pickle=True)

    train_data = []
    test_data = []
    for i in range(X1.shape[0]):
        train_data.append(X1[i])
    for i in range(X2.shape[0]):
        test_data.append(X2[i])

    print('Done !')
    print('Number of beauty train data: ', str(len(train_data)))
    print('Number of beauty test data: ', str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

def random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding, 3)
    new_batch = []
    npad = ((padding, padding), (padding, padding),(0,0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch


def random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def random_rotation(batch, max_angle):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            angle = random.uniform(-max_angle, max_angle)
            batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, reshape=False)
    return batch

# def random_flip_updown(batch):
#     for i in range(len(batch)):
#         if bool(random.getrandbits(1)):
#             batch[i] = np.flipud(batch[i])
#     return batch


# def random_90degrees_rotation(batch, rotations=[0, 1, 2, 3]):
#     for i in range(len(batch)):
#         num_rotations = random.choice(rotations)
#         batch[i] = np.rot90(batch[i], num_rotations)
#     return batch

# def random_blur(batch, sigma_max=5.0):
#     for i in range(len(batch)):
#         if bool(random.getrandbits(1)):
#             sigma = random.uniform(0., sigma_max)
#             batch[i] = scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
#     return batch

def augmentation(batch, img_size):
    batch = random_crop(batch, (img_size, img_size), 10)  #补零裁剪
    #batch = random_blur(batch)  
    batch = random_flip_leftright(batch)   #翻转
    batch = random_rotation(batch, 10)     #旋转
    return batch
