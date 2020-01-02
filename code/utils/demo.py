import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import os
import re
import pickle
import argparse
import tensorflow as tf
import net
import csv

ap = argparse.ArgumentParser()
ap.add_argument("--image",required=True,
               help='Path of the human face image')
ap.add_argument("--usecamera",required = True,
               help='1 means using camera,0 means not using camera')
ap.add_argument('--test_model',required=True,
                help = 'Choose test model')
args=vars(ap.parse_args())
race_dict={0:'White',1:'Black',2:'Asian',3:'Indian',4:'Other'}
def load_model():
    sess=tf.Session()
    x = tf.placeholder(tf.float32,[None,48,48,1])
    y_smile_conv,y_gender_conv,y_glasses_conv,phase_train,keep_prob=net.BKNetModel(x)
    saver = tf.train.Saver(max_to_keep=1)
    print('Restoring existed model')
    saver.restore(sess, '../../save/current/model.ckpt')
    print('OK')

    return sess,x,y_smile_conv,y_gender_conv,y_glasses_conv,phase_train,keep_prob
def load_age_model():
    sess=tf.Session()
    x = tf.placeholder(tf.float32, [None, 48, 48, 1])
    y_age_conv,phase_train,keep_prob = net.AGENETModel(x)
    saver = tf.train.Saver(max_to_keep=1)
    print('Restoring existed model')
    saver.restore(sess, '../../save/current2/model-megaage.ckpt')
    print('OK')
    return sess,x,y_age_conv,phase_train,keep_prob
def load_ethnic_model():
    sess=tf.Session()
    x = tf.placeholder(tf.float32, [None, 48, 48, 1])
    y_ethnic_conv,phase_train,keep_prob = net.ETHNICModel(x)
    saver = tf.train.Saver(max_to_keep=1)
    print('Restoring existed model')
    saver.restore(sess, '../../save/current3/model-ethnic.ckpt')
    print('OK')
    return sess,x,y_ethnic_conv,phase_train,keep_prob
def load_age_eth_model():
    sess = tf.Session()
    x = tf.placeholder(tf.float32, [None, 48, 48, 1])
    y_age_conv,y_ethnic_conv, phase_train, keep_prob = net.AgeEthModel(x)
    saver = tf.train.Saver(max_to_keep=1)
    print('Restoring existed model')
    saver.restore(sess, '../../save/current4/model-age-ethnic.ckpt')
    print('OK')
    return sess, x, y_age_conv,y_ethnic_conv, phase_train, keep_prob

def draw_label(img,x,y,w,h,label,font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1,thickness=2):
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,155,255),2)
    cv2.putText(img,label,(x,y),font,font_scale,(255,255,255),thickness)

def main(sess,x,y_smile_conv,y_gender_conv,y_glasses_conv,phase_train,keep_prob):
    detector = MTCNN()
    if(int(args['usecamera'])==1):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                print("error: failed to capture image")
                return -1

            # detect face and crop face, convert to gray, resize to 48x48
            original_img = img
            cv2.imshow("result", original_img)
            result = detector.detect_faces(original_img)
            if not result:
                cv2.imshow("result", original_img)
                continue
            for face in result:
                face_position = face.get('box')
                x_coordinate = face_position[0]
                y_coordinate = face_position[1]
                w_coordinate = face_position[2]
                h_coordinate = face_position[3]
                img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                if (img.size == 0):
                    cv2.imshow("result", original_img)
                    continue;
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                img = (img - 128) / 255.0
                T = np.zeros([48, 48, 1])
                T[:, :, 0] = img
                test_img = []
                test_img.append(T)
                test_img = np.asarray(test_img)

                predict_y_smile_conv = sess.run(y_smile_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                predict_y_gender_conv = sess.run(y_gender_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                predict_y_glasses_conv = sess.run(y_glasses_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})

                smile_label = "-_-" if np.argmax(predict_y_smile_conv) == 0 else ":)"
                gender_label = "Female" if np.argmax(predict_y_gender_conv) == 0 else "Male"
                glasses_label = 'On Glasses' if np.argmax(predict_y_glasses_conv)==1 else 'No Glasses'

                label = "{}, {}, {}".format(smile_label, gender_label, glasses_label)
                draw_label(original_img, x_coordinate, y_coordinate, w_coordinate, h_coordinate, label)

            cv2.imshow("result", original_img)
            key = cv2.waitKey(1)

            if key == 27:
                break

    else:
        img_list = os.listdir(args['image'])
        with open('label.csv','a') as csv_file:
            writer = csv.writer(csv_file,delimiter = ',')
            for img_name in img_list:
                label_list = []
                original_img = cv2.imread(os.path.join(args['image'],img_name))
                result = detector.detect_faces(original_img)
                if not result:
                    print('can not detect face in the photo')
                    print(img_name)
                    continue
                face_position = result[0].get('box')
                x_coordinate = face_position[0]
                y_coordinate = face_position[1]
                w_coordinate = face_position[2]
                h_coordinate = face_position[3]
                img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                if img.size ==0:
                    print('can not crop the face from the photo')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                img = (img - 128) / 255.0
                T = np.zeros([48, 48, 1])
                T[:, :, 0] = img
                test_img = []
                test_img.append(T)
                test_img = np.asarray(test_img)

                predict_y_smile_conv = sess.run(y_smile_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                predict_y_gender_conv = sess.run(y_gender_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                predict_y_glasses_conv = sess.run(y_glasses_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})

                label_list.append(img_name)
                label_list.append( '-_-' if np.argmax(predict_y_smile_conv)==0 else ':)')
                label_list.append('Female' if np.argmax(predict_y_gender_conv)==0 else 'Male')
                label_list.append('On Glasses' if np.argmax(predict_y_glasses_conv)==1 else 'No Glasses')
                writer.writerow(label_list)

                label = "{}, {}, {}".format(label_list[1], label_list[2], label_list[3])
                draw_label(original_img, x_coordinate, y_coordinate, w_coordinate, h_coordinate, label)

                cv2.imshow("result", original_img)
                key = cv2.waitKey(1)
def main_age(sess,x,y_age_conv,phase_train,keep_prob):
    detector = MTCNN()
    if (int(args['usecamera']) == 1):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                print("error: failed to capture image")
                return -1

            # detect face and crop face, convert to gray, resize to 48x48
            original_img = img
            cv2.imshow("result", original_img)
            result = detector.detect_faces(original_img)
            if not result:
                cv2.imshow("result", original_img)
                continue
            for face in result:
                face_position = face.get('box')
                x_coordinate = face_position[0]
                y_coordinate = face_position[1]
                w_coordinate = face_position[2]
                h_coordinate = face_position[3]
                img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                if (img.size == 0):
                    cv2.imshow("result", original_img)
                    continue;
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                img = (img - 128) / 255.0
                T = np.zeros([48, 48, 1])
                T[:, :, 0] = img
                test_img = []
                test_img.append(T)
                test_img = np.asarray(test_img)
                age_list = [i for i in range(1, 117)]
                predict_y_age_conv = sess.run(y_age_conv,feed_dict={x:test_img,phase_train:False,keep_prob:1})
                age_label = str(int(np.sum(np.multiply(predict_y_age_conv,age_list))))
                label = "{}".format(age_label)
                draw_label(original_img, x_coordinate, y_coordinate, w_coordinate, h_coordinate, label)

            cv2.imshow("result", original_img)
            key = cv2.waitKey(1)

            if key == 27:
                break
    else:
        img_list = os.listdir(args['image'])

        with open('label-imdb-age.csv', 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for subdir in img_list:
                path = os.path.join(args['image'], subdir)
                imgs = os.listdir(path)
                print(subdir)
                for img_name in imgs:
                    label_list = []
                    original_img = cv2.imread(os.path.join(path, img_name))
                    if original_img.all()==None:
                        continue
                    result = detector.detect_faces(original_img)
                    if not result:
                        print('can not detect face in the photo')
                        print(img_name)
                        continue
                    face_position = result[0].get('box')
                    x_coordinate = face_position[0]
                    y_coordinate = face_position[1]
                    w_coordinate = face_position[2]
                    h_coordinate = face_position[3]
                    img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                    if img.size == 0:
                        print('can not crop the face from the photo')
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (48, 48))
                    img = (img - 128) / 255.0
                    T = np.zeros([48, 48, 1])
                    T[:, :, 0] = img
                    test_img = []
                    test_img.append(T)
                    test_img = np.asarray(test_img)
                    age_list = [i for i in range(1, 117)]
                    predict_y_age_conv = sess.run(y_age_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                    age_label = int(np.sum(np.multiply(predict_y_age_conv, age_list)))
                    real_age = int(re.findall('\d+',img_name)[5])-int(re.findall('\d+',img_name)[2])
                    mae = np.abs(real_age-age_label)
                    if(mae<=5):
                        print(os.path.join(subdir,img_name)+' '+str(age_label))
                        label_list.append(os.path.join(subdir,img_name))
                        label_list.append(str(age_label))
                        writer.writerow(label_list)
def main_ethnic(sess,x,y_ethnic_conv,phase_train,keep_prob):
    detector = MTCNN()
    if (int(args['usecamera']) == 1):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                print("error: failed to capture image")
                return -1

            # detect face and crop face, convert to gray, resize to 48x48
            original_img = img
            cv2.imshow("result", original_img)
            result = detector.detect_faces(original_img)
            if not result:
                cv2.imshow("result", original_img)
                continue
            for face in result:
                face_position = face.get('box')
                x_coordinate = face_position[0]
                y_coordinate = face_position[1]
                w_coordinate = face_position[2]
                h_coordinate = face_position[3]
                img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                if (img.size == 0):
                    cv2.imshow("result", original_img)
                    continue;
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                img = (img - 128) / 255.0
                T = np.zeros([48, 48, 1])
                T[:, :, 0] = img
                test_img = []
                test_img.append(T)
                test_img = np.asarray(test_img)
                predict_y_ethnic_conv = sess.run(y_ethnic_conv,feed_dict={x:test_img,phase_train:False,keep_prob:1})
                ethnic_label = np.argmax(predict_y_ethnic_conv)
                label = "{}".format(race_dict[ethnic_label])
                draw_label(original_img, x_coordinate, y_coordinate, w_coordinate, h_coordinate, label)

            cv2.imshow("result", original_img)
            key = cv2.waitKey(1)

            if key == 27:
                break
    else:
        img_list = os.listdir(args['image'])

        with open('label-celeba-race.csv', 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')

            for img_name in img_list:
                label_list = []
                original_img = cv2.imread(os.path.join(args['image'], img_name))
                print(os.path.join(args['image'], img_name))
                if original_img.all()==None:
                    continue
                result = detector.detect_faces(original_img)
                if not result:
                    print('can not detect face in the photo')
                    print(img_name)
                    continue
                face_position = result[0].get('box')
                x_coordinate = face_position[0]
                y_coordinate = face_position[1]
                w_coordinate = face_position[2]
                h_coordinate = face_position[3]
                img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                if img.size == 0:
                    print('can not crop the face from the photo')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                img = (img - 128) / 255.0
                T = np.zeros([48, 48, 1])
                T[:, :, 0] = img
                test_img = []
                test_img.append(T)
                test_img = np.asarray(test_img)
                predict_y_ethnic_conv = sess.run(y_ethnic_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                ethnic_label = np.argmax(predict_y_ethnic_conv)

                print(img_name+' '+str(race_dict[ethnic_label]))
                label_list.append(img_name)
                label_list.append(ethnic_label)
                writer.writerow(label_list)
def main_age_eth(sess,x,y_age_conv,y_ethnic_conv,phase_train,keep_prob):
    detector = MTCNN()
    if (int(args['usecamera']) == 1):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                print("error: failed to capture image")
                return -1

            # detect face and crop face, convert to gray, resize to 48x48
            original_img = img
            cv2.imshow("result", original_img)
            result = detector.detect_faces(original_img)
            if not result:
                cv2.imshow("result", original_img)
                continue
            for face in result:
                face_position = face.get('box')
                x_coordinate = face_position[0]
                y_coordinate = face_position[1]
                w_coordinate = face_position[2]
                h_coordinate = face_position[3]
                img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                if (img.size == 0):
                    cv2.imshow("result", original_img)
                    continue;
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                img = (img - 128) / 255.0
                T = np.zeros([48, 48, 1])
                T[:, :, 0] = img
                test_img = []
                test_img.append(T)
                test_img = np.asarray(test_img)
                predict_y_ethnic_conv = sess.run(y_ethnic_conv,
                                                  feed_dict={x: test_img, phase_train: False, keep_prob: 0.5})
                ethnic_label = np.argmax(predict_y_ethnic_conv)
                age_list = [i for i in range(0, 117)]
                predict_y_age_conv = sess.run(y_age_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 0.5})
                age_label = str(int(np.sum(np.multiply(predict_y_age_conv, age_list))))
                label = "{} {}".format(race_dict[ethnic_label],str(age_label))
                draw_label(original_img, x_coordinate, y_coordinate, w_coordinate, h_coordinate, label)

            cv2.imshow("result", original_img)
            key = cv2.waitKey(1)

            if key == 27:
                break
if __name__ == '__main__':
    if(int(args['test_model'])==1):
        sess, x, y_smile_conv, y_gender_conv, y_glasses_conv, phase_train, keep_prob = load_model()
        main(sess, x, y_smile_conv, y_gender_conv, y_glasses_conv, phase_train, keep_prob)
    elif(int(args['test_model'])==2):
        sess, x, y_age_conv, phase_train, keep_prob = load_age_model()
        main_age(sess, x, y_age_conv, phase_train, keep_prob)
    elif(int(args['test_model'])==3):
        sess,x,y_ethnic_conv,phase_train,keep_prob = load_ethnic_model()
        main_ethnic(sess, x, y_ethnic_conv, phase_train, keep_prob)
    elif(int(args['test_model'])==4):
        sess,x,y_age_conv,y_ethnic_conv,phase_train,keep_prob = load_age_eth_model()
        main_age_eth(sess, x, y_age_conv, y_ethnic_conv, phase_train, keep_prob)










