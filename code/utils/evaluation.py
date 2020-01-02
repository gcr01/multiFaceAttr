import CNNhead_input as CNN2Head_input
#import CNNhead_input
import tensorflow as tf
import numpy as np
import net
from const import *
import os
import datetime


''' PREPARE DATA '''
_, smile_test = CNN2Head_input.getSmileImage()
_, gender_test = CNN2Head_input.getGenderImage()
_, glasses_test = CNN2Head_input.getGlassesImage() #celebA
_,ethnic_test = CNN2Head_input.getEthnicImage() #rfw
_,age_test = CNN2Head_input.getAgeImage() #megaage


def tf_confusion_metrics(predict, real, session, feed_dict, mask):
    predictions = tf.argmax(predict, 1)
    actuals = tf.argmax(real, 1)
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
 
    tp_op = tf.reduce_sum(
        tf.cast( tf.logical_and(tf.equal(actuals, ones_like_actuals),tf.equal(predictions, ones_like_predictions)) , dtype=tf.float32 )* mask
    )
 
    tn_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ), dtype = tf.float32
        ) * mask
    )
 
    fp_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, ones_like_predictions)
          ) , dtype = tf.float32
        ) * mask
    )
 
    fn_op = tf.reduce_sum(
        tf.cast(
          tf.logical_and(
            tf.equal(actuals, ones_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
          ), dtype = tf.float32
        ) * mask
    )
    tp, tn, fp, fn = session.run([tp_op, tn_op, fp_op, fn_op], feed_dict)
    # print('tp:'+str(tp )+'tn:'+str(tn )+'fp:'+str(fp )+'fn:'+str(fn))
    try:
        tpr = float(tp)/(float(tp) + float(fn))
    except ZeroDivisionError:
        recall=0
    else:
        recall=tpr
    # fpr = float(fp)/(float(fp) + float(tn))
    # fnr = float(fn)/(float(tp) + float(fn))
    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))
    # precision = float(tp)/(float(tp) + float(fp))
    # f1_score = (2 * (precision * recall)) / (precision + recall)
    return recall,accuracy

def tf_confusion_metrics_multi(predict_vec, real_vec, session, mask, feed_dict):
    ind = tf.where(mask > 0)
    predictions = tf.argmax(predict_vec, 1)
    actuals = tf.argmax(real_vec, 1)
    actual_eth = tf.gather_nd(actuals, ind)  # extract the useful vector from the whole vector
    pred_eth = tf.gather_nd(predictions, ind)
    cmatrix=tf.confusion_matrix(actual_eth, pred_eth, num_classes = 5)
    confuse_martix, act, pred = session.run([tf.convert_to_tensor(cmatrix),actual_eth,pred_eth],feed_dict)
    
    print(confuse_martix)
    print(act)
    print(pred)

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

def one_hot(index, num_classes):
    tmp = np.zeros(num_classes, dtype=np.float32)
    if(index==-1):   #celebA 1 means positive ;-1 means negative
        tmp[0] = 1.0
    elif(index==1):
        tmp[1]=1.0
    else:
        tmp[index] = 1.0
    return tmp

# def one_hot_age(index, num_classes):
#     assert index>=1 and index <=116
#     tmp = np.zeros(num_classes, dtype=np.float32)
#     tmp[index-1]=1.0
#     return tmp
# #in utk 0 male 1 female ;
# def one_hot_gender(index, num_classes):
#     tmp = np.zeros(num_classes, dtype=np.float32)
#     if(index == 0):   #celebA 1 means positive ;-1 means negative
#         tmp[1] = 1.0
#     elif(index == 1):
#         tmp[0]=1.0
#     return tmp

# def eval_age():
#     with tf.Session() as sess:
#         x, y_= net.Input_age()
#         y_age_conv, phase_train, keep_prob = net.AGENETModel(x)
#         age_loss, l2_loss, loss = net.age_loss(y_age_conv, y_)
#         if tf.test.gpu_device_name():
#             print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
#         else:
#             print("Please install GPU version of TF")
#         #y_age = tf.get_collection('y_age')[0]
#         y_age_expectation = tf.cast(tf.argmax(y_age_conv,1)+1,dtype=tf.float32)
#         y_age_ = tf.cast(tf.argmax(y_,1)+1,dtype=tf.float32) #real ages vector
#         age_mae = tf.reduce_mean(tf.abs(tf.subtract(y_age_,y_age_expectation)))
#         test_data = []

#         for i in range(len(age_test)):
#             img = (age_test[i][0] - 128) / 255.0
#             label = (int)(age_test[i][1])
#             test_data.append((img, one_hot_age(label, 116)))
#         np.random.shuffle(test_data)
#         saver = tf.train.Saver(max_to_keep=1)
#         print('Starting eval,restoring existed age model')
#         saver.restore(sess, './save/current2/utk/model-age-utk.ckpt')
#         print('OK')

#         test_img = []
#         test_label = []

#         for i in range(len(test_data)):
#             test_img.append(test_data[i][0])
#             test_label.append(test_data[i][1])

#         number_batch = len(test_data) // BATCH_SIZE

#         print("length of age test data :"+str(len(test_data)))
#         #print("batches:"+str(number_batch))
#         #test_len=len(test_data)
#         avg_mae = []
#         avg_ttl = []

#         start_time=datetime.datetime.now()
#         print('Start time: ' + str(start_time ))

#         for batch in range(number_batch):
#             top = batch * BATCH_SIZE
#             bot = min((batch + 1) * BATCH_SIZE, len(test_data))
#             batch_img = np.asarray(test_img[top:bot])
#             batch_label = np.asarray(test_label[top:bot])
#             batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 1))
            
#             age_test_mae = sess.run(age_mae,
#                                         feed_dict={x: batch_img, y_: batch_label, phase_train: False, keep_prob: 1.0})
#             ttl,_,_= sess.run([loss,l2_loss,age_loss],
#                                           feed_dict={x:batch_img,y_:batch_label,phase_train:False,keep_prob:1.0}) 
#             # print('batch:' + str(batch) + ' total loss:' + str(ttl) + ' mae of this batch:' + str(age_test_mae))
#             avg_ttl.append(ttl)
#             avg_mae.append(age_test_mae)

#         avg_ttl = np.average(avg_ttl)
#         avg_mae = np.average(avg_mae)
        
#         finish_time=datetime.datetime.now()
#         print('Start time: ' + str(start_time ))
#         print('finish time: ' + str(finish_time ))

#         time_duration=(finish_time-start_time).seconds+((finish_time-start_time).microseconds)/1000000
#         print(str(time_duration))
#         fps=len(test_data)/time_duration
#         print('FPS: '+str(fps))
#         print('Model trained by: '+agemodelName)
#         print('TestingSet: '+agetestName)
#         print('Age Test Mae this time:' + str(avg_mae))
#         print('Total test Loss this time:' +str(avg_ttl))
#         print('\n')



def eval_multi():
    with tf.Session() as sess:
        x, y_, mask = net.Input()
        y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv, phase_train, keep_prob = net.BKNetModel(x)

        smile_loss, gender_loss, glasses_loss, ethnic_loss, age_loss, l2_loss, loss = net.selective_loss(y_smile_conv, y_gender_conv,
                                                                                      y_glasses_conv, y_ethnic_conv, y_age_conv, y_,mask)
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")


        smile_mask = tf.get_collection('smile_mask')[0]
        gender_mask = tf.get_collection('gender_mask')[0]
        glasses_mask = tf.get_collection('glasses_mask')[0]
        ethnic_mask = tf.get_collection('ethnic_mask')[0]
        age_mask = tf.get_collection('age_mask')[0]

        y_smile = tf.get_collection('y_smile')[0]
        y_gender = tf.get_collection('y_gender')[0]
        y_glasses = tf.get_collection('y_glasses')[0]
        y_ethnic = tf.get_collection('y_ethnic')[0]
        y_age = tf.get_collection('y_age')[0]

        # smile_correct_prediction = tf.equal(tf.argmax(y_smile_conv, 1), tf.argmax(y_smile, 1))  
        # gender_correct_prediction = tf.equal(tf.argmax(y_gender_conv, 1), tf.argmax(y_gender, 1))
        # glasses_correct_prediction = tf.equal(tf.argmax(y_glasses_conv, 1), tf.argmax(y_glasses, 1))
        # ethnic_correct_prediction = tf.equal(tf.argmax(y_ethnic_conv, 1), tf.argmax(y_ethnic, 1))
        # age_correct_prediction = tf.equal(tf.argmax(y_age_conv, 1), tf.argmax(y_age, 1))

        # smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32) * smile_mask)
        # gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32) * gender_mask)
        # glasses_true_pred = tf.reduce_sum(tf.cast(glasses_correct_prediction, dtype=tf.float32) * glasses_mask)
        # ethnic_true_pred = tf.reduce_sum(tf.cast(ethnic_correct_prediction, dtype=tf.float32) * ethnic_mask)
        # age_true_pred = tf.reduce_sum(tf.cast(age_correct_prediction, dtype=tf.float32) * age_mask)

        test_data = []

        # Mask: Smile -> 0, Gender -> 1, Glasses -> 2, Ethnic -> 3, Age -> 4.
        for i in range(len(smile_test)):
            img = (smile_test[i][0]*1.0 - 128) / 255.0  # 1.from [0,255] to [-128,127] 2.from [-128,127] to [-0.5,0.5]
            label = (int)(smile_test[i][1])
            test_data.append((img, one_hot(label, 5), 0.0))
        for i in range(len(gender_test)):
            img = (gender_test[i][0]*1.0 - 128) / 255.0
            label = (int)(gender_test[i][1])
            test_data.append((img, one_hot(label, 5), 1.0))
        for i in range(len(glasses_test)):
            img = (glasses_test[i][0]*1.0 - 128) / 255.0
            label = (int)(glasses_test[i][1])
            test_data.append((img, one_hot(label, 5), 2.0))
        for i in range(len(ethnic_test)):
            img = (ethnic_test[i][0]*1.0 - 128) / 255.0
            label = (int)(ethnic_test[i][1])
            test_data.append((img, one_hot(label, 5), 3.0))
        for i in range(len(age_test)):
            img = (age_test[i][0]*1.0 - 128) / 255.0
            label = (int)(age_test[i][1])
            test_data.append((img, one_hot(label, 5), 4.0))

        np.random.shuffle(test_data)
        saver = tf.train.Saver(max_to_keep=1)
        print('Restoring existed model')
        saver.restore(sess, './save/current/model.ckpt')
        print('OK')

        test_img = []
        test_label = []
        test_mask = []

        avg_smile_recall = []
        avg_gender_recall = []
        avg_glasses_recall = []
        # avg_ethnic_recall = []
        # avg_age_recall = []

        '''
        to optimize
        use two arrays, 
        one to store the history res, 
        one to store current res,
        add them up each iteration and finaly we get a array with whole res
        it divide by batches numbers result in the avg array '''

        avg_white_recall = []
        avg_black_recall = []
        avg_asian_recall = []
        avg_indian_recall = []
        avg_other_recall = []

        avg_child_recall = []
        avg_teen_recall = []
        avg_young_recall = []
        avg_middle_recall = []
        avg_old_recall = []

        avg_smile_acc = []
        avg_gender_acc = []
        avg_glasses_acc = []
        avg_ethnic_acc = []
        avg_age_acc = []

        for i in range(len(test_data)):
            test_img.append(test_data[i][0])
            test_label.append(test_data[i][1])
            test_mask.append(test_data[i][2])

        number_batch = len(test_data) // BATCH_SIZE

        # smile_nb_true_pred = 0
        # gender_nb_true_pred = 0
        # glasses_nb_true_pred = 0
        # ethnic_nb_true_pred = 0


        # smile_nb_test = 0
        # gender_nb_test = 0
        # glasses_nb_test = 0
        # ethnic_nb_test = 0
        print("length of test data :"+str(len(test_data)))
        print("batches:"+str(number_batch))

        start_time=datetime.datetime.now()
        print('Start time: ' + str(start_time ))

        for batch in range(number_batch):
            top = batch * BATCH_SIZE
            bot = min((batch + 1) * BATCH_SIZE, len(test_data))
            batch_img = np.asarray(test_img[top:bot])
            batch_label = np.asarray(test_label[top:bot])
            batch_mask = np.asarray(test_mask[top:bot])
            # for i in range(BATCH_SIZE):
            #     if batch_mask[i] == 0.0:
            #         smile_nb_test += 1
            #     elif batch_mask[i] == 1.0:
            #         gender_nb_test += 1
            #     elif batch_mask[i] == 2.0:
            #         glasses_nb_test += 1
            #     else:
            #         ethnic_nb_test += 1
            batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 3))
            
            # smile_nb_true_pred += sess.run(smile_true_pred, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
            #                                                            phase_train: False,
            #                                                            keep_prob: 1})
            # gender_nb_true_pred += sess.run(gender_true_pred,
            #                                 feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
            #                                            phase_train: False,
            #                                            keep_prob: 1})
            # glasses_nb_true_pred += sess.run(glasses_true_pred,
            #                                  feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
            #                                             phase_train: False,
            #                                             keep_prob: 1})
            # ethnic_nb_true_pred += sess.run(ethnic_true_pred,
            #                                  feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
            #                                             phase_train: False,
            #                                             keep_prob: 1})
            smile_recall,smile_acc = tf_confusion_metrics(predict=y_smile_conv, real=y_smile, session=sess, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                            phase_train:False,
                                                            keep_prob: 1.0}, mask = smile_mask)
            gender_recall,gender_acc = tf_confusion_metrics(predict=y_gender_conv, real=y_gender, session=sess, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                            phase_train:False,
                                                            keep_prob: 1.0}, mask = gender_mask)
            glasses_recall,glasses_acc = tf_confusion_metrics(predict=y_glasses_conv, real=y_glasses, session=sess, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                            phase_train:False,
                                                            keep_prob: 1.0}, mask = glasses_mask)
            matrix_ethnic = tf_confusion_metrics_multi(y_ethnic_conv, y_ethnic, sess,ethnic_mask, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                        phase_train:False,
                                                        keep_prob: 1.0})
            matrix_age = tf_confusion_metrics_multi(y_age_conv, y_age, sess, age_mask, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                        phase_train:False,
                                                        keep_prob: 1.0})

            ethnic_acc, _, white_recall, black_recall, asian_recall, indian_recall, other_recall = cal_recall_acc_multi(matrix_ethnic)
            age_acc, _, child_recall, teen_recall, young_recall, middle_recall, old_recall = cal_recall_acc_multi(matrix_age)

            avg_smile_recall.append(smile_recall)
            avg_gender_recall.append(gender_recall)
            avg_glasses_recall.append(glasses_recall)
            

            avg_smile_acc.append(smile_acc)
            avg_gender_acc.append(gender_acc)
            avg_glasses_acc.append(glasses_acc)
            avg_ethnic_acc.append(ethnic_acc)
            avg_age_acc.append(age_acc)

            avg_white_recall.append(white_recall)
            avg_black_recall.append(black_recall)
            avg_asian_recall.append(asian_recall)
            avg_indian_recall.append(indian_recall)
            avg_other_recall.append(other_recall)

            avg_child_recall.append(child_recall)
            avg_teen_recall.append(teen_recall)
            avg_young_recall.append(young_recall)
            avg_middle_recall.append(middle_recall)
            avg_old_recall.append(old_recall)
            print('batch' + str(batch) + ' white_recall: ' + str(white_recall * 100) + '%')
            print('batch' + str(batch) + ' black_recall: ' + str(black_recall * 100) + '%')
            print('batch' + str(batch) + ' asian_recall: ' + str(asian_recall * 100) + '%')
            print('batch' + str(batch) + ' indian_recall: ' + str(indian_recall * 100) + '%')
            print('batch' + str(batch) + ' other_recall: ' + str(other_recall * 100) + '%')
            print('\n')
            print('batch' + str(batch) + ' chilid_recall: ' + str(child_recall * 100) + '%')
            print('batch' + str(batch) + ' teen_recall: ' + str(teen_recall * 100) + '%')
            print('batch' + str(batch) + ' young_recall: ' + str(young_recall * 100) + '%')
            print('batch' + str(batch) + ' middle_recall: ' + str(middle_recall * 100) + '%')
            print('batch' + str(batch) + ' old_recall: ' + str(old_recall * 100) + '%')
            print('\n')
        finish_time=datetime.datetime.now()
        print('Start time: ' + str(start_time ))
        print('finish time: ' + str(finish_time ))
        time_duration=(finish_time-start_time).seconds+((finish_time-start_time).microseconds)/1000000
        print(str(time_duration))
        fps=len(test_data)/time_duration
        print('FPS: '+str(fps))
        # print("smile test number :" + str(smile_nb_test))
        # print("gender test number :" + str(gender_nb_test))
        # print("glasses test number :" + str(glasses_nb_test))
        # print("ethnic test number :" + str(ethnic_nb_test))
        # smile_test_accuracy = smile_nb_true_pred * 1.0 / smile_nb_test
        # gender_test_accuracy = gender_nb_true_pred * 1.0 / gender_nb_test
        # glasses_test_accuracy = glasses_nb_true_pred * 1.0 / glasses_nb_test
        
        avg_smile_recall = np.average(avg_smile_recall)
        avg_gender_recall = np.average(avg_gender_recall)
        avg_glasses_recall = np.average(avg_glasses_recall)
        #avg_ethnic_recall = np.average(avg_ethnic_recall)
        #avg_age_recall = np.average(avg_age_recall)

        avg_smile_acc = np.average(avg_smile_acc)
        avg_gender_acc = np.average(avg_gender_acc)
        avg_glasses_acc = np.average(avg_glasses_acc)
        avg_ethnic_acc = np.average(avg_ethnic_acc)
        avg_age_acc = np.average(avg_age_acc)
        
        avg_white_recall = np.average(avg_white_recall)
        avg_black_recall = np.average(avg_black_recall)
        avg_asian_recall = np.average(avg_asian_recall)
        avg_indian_recall = np.average(avg_indian_recall)
        avg_other_recall = np.average(avg_other_recall)
        avg_ethnic_recall = (avg_white_recall + avg_black_recall + avg_asian_recall + avg_indian_recall + avg_other_recall)/5

        avg_child_recall = np.average(avg_child_recall)
        avg_teen_recall = np.average(avg_teen_recall)
        avg_young_recall = np.average(avg_young_recall)
        avg_middle_recall = np.average(avg_middle_recall)
        avg_old_recall = np.average(avg_old_recall)
        avg_age_recall = (avg_child_recall + avg_teen_recall + avg_young_recall + avg_middle_recall + avg_old_recall)/5
        
    
        print('Smile task test accuracy: ' + str(avg_smile_acc * 100)+'%')
        print('Gender task test accuracy: ' + str(avg_gender_acc * 100)+'%')
        print('Glasses task test accuracy: ' + str(avg_glasses_acc * 100)+'%')
        print('Ethnic task test accuracy: ' + str(avg_ethnic_acc * 100)+'%')
        print('Age task test accuracy: ' + str(avg_age_acc * 100)+'%')
        print('\n')

        print('Smile task test recall: ' + str(avg_smile_recall * 100)+'%')
        print('Gender task test recall: ' + str(avg_gender_recall * 100)+'%')
        print('Glasses task test recall: ' + str(avg_glasses_recall * 100)+'%')
        print('Ethnic task test recall: ' + str(avg_ethnic_recall * 100)+'%')
        print('Age task test recall: ' + str(avg_age_recall * 100)+'%')
        print('\n')
        print('White test recall: ' + str(avg_white_recall * 100)+'%')
        print('Balck test recall: ' + str(avg_black_recall * 100)+'%')
        print('Asian test recall: ' + str(avg_asian_recall * 100)+'%')
        print('Indian test recall: ' + str(avg_indian_recall * 100)+'%')
        print('Other test recall: ' + str(avg_other_recall * 100)+'%')
        print('\n')
        print('Child test recall: ' + str(avg_child_recall * 100)+'%')
        print('Teen test recall: ' + str(avg_teen_recall * 100)+'%')
        print('Young test recall: ' + str(avg_young_recall * 100)+'%')
        print('Middle test recall: ' + str(avg_middle_recall * 100)+'%')
        print('Old test recall: ' + str(avg_old_recall * 100)+'%')
        print('\n')

if __name__ =='__main__':
    eval_multi()



