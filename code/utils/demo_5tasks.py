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
args=vars(ap.parse_args())

race_dict={0:'W', 1:'B', 2:'Y', 3:'I', 4:'O'}
age_dict = {0:'C', 1:'T', 2:'Y', 3:'M', 4:'O'}

def load_model():
    sess = tf.Session()
    x = tf.placeholder(tf.float32,[None,48,48,3])
    y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv, phase_train, keep_prob = net.BKNetModel(x)
    saver = tf.train.Saver(max_to_keep=1)
    print('Restoring existed model')
    saver.restore(sess, './save/current/model.ckpt')
    print('OK')

    return sess, x, y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv, phase_train, keep_prob

def draw_text(img, point, text, drawType="simple"):
    fontScale = 0.6
    thickness = 1
    text_thickness = 1
    bg_color = (255, 0, 0)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
        # draw score value
        cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale,
                    (255, 255, 255), text_thickness, 8)
    elif drawType == "simple":
        # cv2.putText(img, '%d' % (text), point, fontFace, 0.5, (255, 0, 0))
        cv2.putText(img, text, tuple(point), fontFace, fontScale, (0, 0, 255), thickness)
    return img
 
def draw_text_line(img, point, point_, text_line: str, drawType="simple"):
    fontScale = 0.4
    thickness = 5
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, point, point_, (0, 155, 255), 2)
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split("\n")
    # text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    for i, text in enumerate(text_line):
        if text:
            draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            img = draw_text(img, draw_point, text, drawType)
    return img
 

def draw_label(i, img, x, y, w, h, label, font = cv2.FONT_HERSHEY_PLAIN, font_scale = 2, thickness = 1):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 155, 255), 2)
    cv2.putText(img, label, (x, y), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(img, str(i+1), (x-10, y), font, 1, (0, 155, 255), thickness)


def main(sess, x, y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv, phase_train, keep_prob):
    detector = MTCNN(min_face_size = 48)
    if(int(args['usecamera'])==1):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                print("error: failed to capture image")
                return -1

            # detect face and crop face, resize to 48x48
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
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                #print(img.shape)
                img = (img*1.0 - 128) / 255.0
                T = np.zeros([48, 48, 3])
                T[:, :, :] = img
                test_img = []
                test_img.append(T)
                test_img = np.asarray(test_img)
                predict_y_smile_conv, predict_y_gender_conv, predict_y_glasses_conv, predict_y_ethnic_conv ,predict_y_age_conv = sess.run([y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv],feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                smile_label = "-_-" if np.argmax(predict_y_smile_conv) == 0 else ":)"
                gender_label = "Female" if np.argmax(predict_y_gender_conv) == 0 else "Male"
                glasses_label = 'On Glasses' if np.argmax(predict_y_glasses_conv)==1 else 'No Glasses'
                ethnic_label = np.argmax(predict_y_ethnic_conv)
                age_label = np.argmax(predict_y_age_conv)

                label = "{}, {}, {}, {}, {}".format(smile_label, gender_label, glasses_label, race_dict[ethnic_label], age_dict[age_label])
                # print(label)
                draw_label(original_img, x_coordinate, y_coordinate, w_coordinate, h_coordinate, label)

            cv2.imshow("result", original_img)
            key = cv2.waitKey(1)
            if key == 27:
                break

    else:
        img_list = os.listdir(args['image'])
        # with open('label.','a') as csv_file:
        #     writer = csv.writer(csv_file,delimiter = ',')
        for img_name in img_list:
            label_list = []
            pts_list = []
            cordinate = []
            path = os.path.join(args['image'],img_name)
            with open( 'E:/gucongrong/FaceAttributeDetection/test_label/' + img_name.split('.')[0]+'.txt', 'a' ) as txt_file:
                original_img = cv2.imread(path)
                height=original_img.shape[0]
                width=original_img.shape[1]
                if original_img is None:
                    print(img_name+' is broken')
                    os.remove(os.path.join(path))
                    continue
                result = detector.detect_faces(original_img)
                if not result:
                    print('can not detect face in '+img_name)
                    os.remove(os.path.join(path))
                    continue
                for i in range(len(result)):
                    face_position = result[i].get('box')
                    x_coordinate = face_position[0]
                    y_coordinate = face_position[1]
                    w_coordinate = face_position[2]
                    h_coordinate = face_position[3]
                    cordinate.append((x_coordinate,y_coordinate,w_coordinate,h_coordinate))
                    pts_ = result[i].get('keypoints')
                    pts = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (pts_['left_eye'][0], pts_['left_eye'][1], 
                                                             pts_['right_eye'][0], pts_['right_eye'][1], 
                                                             pts_['nose'][0], pts_['nose'][1], 
                                                             pts_['mouth_left'][0], pts_['mouth_left'][1], 
                                                             pts_['mouth_right'][0], pts_['mouth_right'][1]
                                                             )
                    pts_list.append(pts)
                    if x_coordinate < 0:
                        x_coordinate = 0
                    if y_coordinate < 0:
                        y_coordinate = 0
                    if w_coordinate > width:
                        w_coordinate = width
                    if h_coordinate > height:
                        h_coordinate = height
                    img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                    if img.size ==0:
                        print('can not crop the face from '+img_name)
                        os.remove(os.path.join(path))
                        continue
                    img = cv2.resize(img, (48, 48))
                    img = (img*1.0 - 128) / 255.0
                    test_img = np.asarray([img])
                    predict_y_smile_conv, predict_y_gender_conv, predict_y_glasses_conv, predict_y_ethnic_conv ,predict_y_age_conv = sess.run([y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv], 
                                                                                                                                                            feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                    
                    ethnic_label = int(np.argmax(predict_y_ethnic_conv))
                    age_label = int(np.argmax(predict_y_age_conv))

                    label_list.append(img_name)
                    label_list.append( 'N' if np.argmax(predict_y_smile_conv)==0 else 'Y')
                    label_list.append('F' if np.argmax(predict_y_gender_conv)==0 else 'M')
                    label_list.append('N' if np.argmax(predict_y_glasses_conv)==0 else 'Y')
                    label_list.append(race_dict[ethnic_label])
                    label_list.append(age_dict[age_label])

                    # label = "{}\n{}\n{}\n{}\n{}".format(label_list[5], label_list[2], label_list[3], label_list[4], label_list[1])
                    label = "{}{}{}{}{}".format(label_list[5], label_list[2], label_list[3], label_list[4], label_list[1])
                    # draw_text_line(original_img, (x_coordinate, y_coordinate), (x_coordinate + w_coordinate, y_coordinate + h_coordinate),label)
                    draw_label(i, original_img, x_coordinate, y_coordinate, w_coordinate, h_coordinate, label)

                txt_file.write('total_number : ' + str(len(result)) + '\n' )
                for i in range(len(result)):
                    txt_file.write('id : ' + str(i+1) + '\n')
                    txt_file.write('face_rectangle : ' + '%d,%d,%d,%d' % cordinate[i] +'\n')
                    txt_file.write('face_pts : ' + pts_list[i] + '\n')
                    txt_file.write('age : ' + label_list[5] + '\n')
                    txt_file.write('gender : ' + label_list[2] + '\n')
                    txt_file.write('hasGlass : ' + label_list[3] + '\n')
                    txt_file.write('race : ' + label_list[4] + '\n')
                    txt_file.write('Smile : ' + label_list[1] + '\n')
            cv2.imwrite("./test_res/%s" % img_name, original_img)


if __name__ == '__main__':
    sess, x, y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv, phase_train, keep_prob = load_model()
    main(sess, x, y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv, phase_train, keep_prob)
    










