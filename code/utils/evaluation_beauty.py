import CNNhead_input as CNN2Head_input
#import CNNhead_input
import tensorflow as tf
import numpy as np
import net
from const import *
import os
import datetime


''' PREPARE DATA '''
_,beauty_test = CNN2Head_input.getBeautyImage()

def tf_confusion_metrics_multi(predict_vec, real_vec, session, feed_dict):
    predictions = tf.argmax(predict_vec, 1)
    actuals = tf.argmax(real_vec, 1)
    cmatrix=tf.confusion_matrix(actuals, predictions, num_classes = 5)
    confuse_martix = session.run(tf.convert_to_tensor(cmatrix),feed_dict)

    return confuse_martix

def cal_recall_acc_multi(cmatrix):
    if np.sum(cmatrix[0]) == 0:
        zero_recall = 0.87
    else:
        zero_recall = cmatrix[0][0] / np.sum(cmatrix[0])

    if np.sum(cmatrix[1]) == 0:
        one_recall = 0.87
    else:
        one_recall = cmatrix[1][1] / np.sum(cmatrix[1])

    if np.sum(cmatrix[2]) == 0:
        two_recall = 0.87
    else:
        two_recall = cmatrix[2][2] / np.sum(cmatrix[2])

    if np.sum(cmatrix[3]) == 0:
        three_recall = 0.87
    else:
        three_recall = cmatrix[3][3] / np.sum(cmatrix[3])

    if np.sum(cmatrix[4]) == 0:
        four_recall = 0.87
    else:
        four_recall = cmatrix[4][4] / np.sum(cmatrix[4])
    tpre = cmatrix[0][0] + cmatrix[1][1] + cmatrix[2][2] + cmatrix[3][3] + cmatrix[4][4]
    nb = np.sum(cmatrix[0]) + np.sum(cmatrix[1]) + np.sum(cmatrix[2])+ np.sum(cmatrix[3]) + np.sum(cmatrix[4])
    acc = tpre/nb
    recall= (zero_recall + one_recall + two_recall + three_recall + four_recall ) / 5
    return acc, recall, zero_recall, one_recall, two_recall, three_recall, four_recall

#[1,2,3,4,5] label 1point stored in idx0 of an array
def one_hot(index, num_classes):
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index-1] = 1.0
    return tmp



def eval_beauty():
    with tf.Session() as sess:
        x, y_= net.Input_beauty()
        y_beauty_conv, phase_train, keep_prob = net.BeautyNETModel(x)
        beauty_loss, l2_loss, loss = net.beauty_loss(y_beauty_conv, y_)
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")
        y_beauty = tf.get_collection('y_beauty')[0]
        y_beauty_expectation = tf.cast(tf.argmax(y_beauty_conv,1)+1,dtype=tf.float32)
        y_beauty_ = tf.cast(tf.argmax(y_beauty,1)+1,dtype=tf.float32) #real score vector
        beauty_mae = tf.reduce_mean(tf.abs(tf.subtract(y_beauty_,y_beauty_expectation)))
        test_data = []

        for i in range(len(beauty_test)):
            img = (beauty_test[i][0]*1.0 - 128) / 255.0
            label = (int)(beauty_test[i][1])
            test_data.append((img, one_hot(label, 5)))
        np.random.shuffle(test_data)
        saver = tf.train.Saver(max_to_keep=1)
        print('Starting eval,restoring existed beauty model')
        saver.restore(sess, './save/current3/model_beauty.ckpt')
        print('OK')

        test_img = []
        test_label = []

        for i in range(len(test_data)):
            test_img.append(test_data[i][0])
            test_label.append(test_data[i][1])

        number_batch = len(test_data) // BATCH_SIZE

        print("length of beauty test data :"+str(len(test_data)))
        
        avg_mae = []
        avg_ttl = []
        avg_beauty_acc = []
        avg_one_recall = []
        avg_two_recall = []
        avg_three_recall = []
        avg_four_recall = []
        avg_five_recall = []

        start_time=datetime.datetime.now()
        print('Start time: ' + str(start_time ))

        for batch in range(number_batch):
            top = batch * BATCH_SIZE
            bot = min((batch + 1) * BATCH_SIZE, len(test_data))
            batch_img = np.asarray(test_img[top:bot])
            batch_label = np.asarray(test_label[top:bot])
            batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 3))
            
            beauty_test_mae = sess.run(beauty_mae,
                                        feed_dict={x: batch_img, y_: batch_label, phase_train: False, keep_prob: 1.0})
            ttl,_,_= sess.run([loss,l2_loss,beauty_loss],
                                          feed_dict={x:batch_img,y_:batch_label,phase_train:False,keep_prob:1.0})

            matrix_beauty = tf_confusion_metrics_multi(y_beauty_conv, y_beauty, sess, feed_dict={x: batch_img, y_: batch_label,
                                                        phase_train:False,
                                                        keep_prob: 1.0})

            beauty_acc, _, one_recall, two_recall, three_recall, four_recall, five_recall = cal_recall_acc_multi(matrix_beauty)
            
            avg_ttl.append(ttl)
            avg_mae.append(beauty_test_mae)
            avg_beauty_acc.append(beauty_acc)
            avg_one_recall.append(one_recall)
            avg_two_recall.append(two_recall)
            avg_three_recall.append(three_recall)
            avg_four_recall.append(four_recall)
            avg_five_recall.append(five_recall)

            print('batch' + str(batch) + ' Beauty Acc: ' + str(beauty_acc * 100) + '%')
            print('batch' + str(batch) + ' one_recall: ' + str(one_recall * 100) + '%')
            print('batch' + str(batch) + ' two_recall: ' + str(two_recall * 100) + '%')
            print('batch' + str(batch) + ' three_recall: ' + str(three_recall * 100) + '%')
            print('batch' + str(batch) + ' four_recall: ' + str(four_recall * 100) + '%')
            print('batch' + str(batch) + ' five_recall: ' + str(five_recall * 100) + '%')
            print('\n')

        avg_ttl = np.average(avg_ttl)
        avg_mae = np.average(avg_mae)
        avg_beauty_acc = np.average(avg_beauty_acc)
        avg_one_recall = np.average(avg_one_recall)
        avg_two_recall = np.average(avg_two_recall)
        avg_three_recall = np.average(avg_three_recall)
        avg_four_recall = np.average(avg_four_recall)
        avg_five_recall = np.average(avg_five_recall)
        avg_beauty_recall = (avg_one_recall + avg_two_recall + avg_three_recall + avg_four_recall + avg_five_recall)/5
        
        finish_time=datetime.datetime.now()
        print('Start time: ' + str(start_time ))
        print('finish time: ' + str(finish_time ))

        time_duration=(finish_time-start_time).seconds+((finish_time-start_time).microseconds)/1000000
        print(str(time_duration))
        fps=len(test_data)/time_duration
        print('FPS: '+str(fps))

        
        print('Beauty Test Mae this time:' + str(avg_mae))
        print('Total test Loss this time:' +str(avg_ttl))
        print('Beauty task test recall: ' + str(avg_beauty_recall * 100)+'%')
        print('\n')
        print('One score test recall: ' + str(avg_one_recall * 100)+'%')
        print('Two score test recall: ' + str(avg_two_recall * 100)+'%')
        print('Four score test recall: ' + str(avg_three_recall * 100)+'%')
        print('Three score test recall: ' + str(avg_four_recall * 100)+'%')
        print('Five score test recall: ' + str(avg_five_recall * 100)+'%')
        print('Beauty Acc: ' + str(avg_beauty_acc * 100)+'%')
        print('\n')

if __name__ =='__main__':
    eval_beauty()



