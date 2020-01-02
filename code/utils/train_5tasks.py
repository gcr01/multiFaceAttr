import CNNhead_input as CNN2Head_input
#import CNNhead
import os
import tensorflow as tf
import numpy as np
import net
from const import *
import random
import argparse

##########train 5 model at one time : gender,race,glass,smile,age############

# ap = argparse.ArgumentParser()
# ap.add_argument('--train_model',required=True,
#                 help = 'Choose train function')
# args = vars(ap.parse_args())

def one_hot(index, num_classes):
    tmp = np.zeros(num_classes, dtype=np.float32)
    if(index==-1):   #celebA 1 means positive ;-1 means negative
        tmp[0] = 1.0
    elif(index==1):
        tmp[1]=1.0
    else:
        tmp[index] = 1.0
    return tmp
# #in utk 0 male.1 female ;
# def one_hot_gender(index, num_classes):
#     tmp = np.zeros(num_classes, dtype=np.float32)
#     if(index == 0):   #celebA 1 means positive ;-1 means negative
#         tmp[1] = 1.0
#     elif(index == 1):
#         tmp[0]=1.0
#     return tmp

def train_multi_task():
    ''' PREPARE DATA '''
    smile_train, smile_test = CNN2Head_input.getSmileImage()
    gender_train, gender_test = CNN2Head_input.getGenderImage()
    afadgender_train, afadgender_test = CNN2Head_input.getAfadGenderImage()
    glasses_train, glasses_test = CNN2Head_input.getGlassesImage()
    ethnic_train, ethnic_test = CNN2Head_input.getEthnicImage()
    age_train, age_test = CNN2Head_input.getAgeImage()
    afadage_train, _ = CNN2Head_input.getAfadAgeImage()
    with tf.Session() as sess:
        global_step = tf.contrib.framework.get_or_create_global_step()
        x, y_, mask = net.Input()
        y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv, phase_train, keep_prob = net.BKNetModel(x)
        smile_loss, gender_loss, glasses_loss, ethnic_loss, age_loss, l2_loss, loss = net.selective_loss(y_smile_conv, y_gender_conv,
                                                                                     y_glasses_conv, y_ethnic_conv, y_age_conv, y_, mask)
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

        train_step = net.train_op(loss, global_step)

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

        smile_correct_prediction = tf.equal(tf.argmax(y_smile_conv, 1), tf.argmax(y_smile, 1))
        gender_correct_prediction = tf.equal(tf.argmax(y_gender_conv, 1), tf.argmax(y_gender, 1))
        glasses_correct_prediction = tf.equal(tf.argmax(y_glasses_conv, 1), tf.argmax(y_glasses, 1))
        ethnic_correct_prediction = tf.equal(tf.argmax(y_ethnic_conv, 1), tf.argmax(y_ethnic, 1))
        age_correct_prediction = tf.equal(tf.argmax(y_age_conv, 1), tf.argmax(y_age, 1))

        smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32) * smile_mask)
        gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32) * gender_mask)
        glasses_true_pred = tf.reduce_sum(tf.cast(glasses_correct_prediction, dtype=tf.float32) * glasses_mask)
        ethnic_true_pred = tf.reduce_sum(tf.cast(ethnic_correct_prediction, dtype=tf.float32) * ethnic_mask)
        age_true_pred = tf.reduce_sum(tf.cast(age_correct_prediction, dtype=tf.float32) * age_mask)


        # Mask: Smile -> 0, Gender -> 1, Glasses -> 2, Ethnic -> 3, Age -> 4
        train_data = []
        ###smile data###
        for i in range(len(smile_train)):
            img = (smile_train[i][0]*1.0 - 128) / 255.0
            label = (int)(smile_train[i][1])
            train_data.append((img, one_hot(label, 5), 0.0))
        ###gender data###
        for i in range(len(gender_train)):
            img = (gender_train[i][0]*1.0 - 128) / 255.0
            label = (int)(gender_train[i][1])
            train_data.append((img, one_hot(label, 5), 1.0))
        for i in range(len(afadgender_train)):
            img = (afadgender_train[i][0]*1.0 - 128) / 255.0
            label = (int)(afadgender_train[i][1])
            train_data.append((img, one_hot(label, 5), 1.0))
        ###glasses data###
        for i in range(len(glasses_train)):
            img = (glasses_train[i][0]*1.0 - 128) / 255.0
            label = (int)(glasses_train[i][1])
            train_data.append((img, one_hot(label, 5), 2.0))
        ###ethnic data###
        for i in range(len(ethnic_train)):
            img = (ethnic_train[i][0]*1.0 - 128) / 255.0
            label = (int)(ethnic_train[i][1])
            train_data.append((img, one_hot(label, 5), 3.0))
        ###age data###
        for i in range(len(age_train)):
            img = (age_train[i][0]*1.0 - 128) / 255.0
            label = (int)(age_train[i][1])
            train_data.append((img, one_hot(label, 5), 4.0))
        for i in range(len(afadage_train)):
            img = (afadage_train[i][0]*1.0 - 128) / 255.0
            label = (int)(afadage_train[i][1])
            train_data.append((img, one_hot(label, 5), 4.0))


        test_data = []
        ###smile data###
        for i in range(len(smile_test)):
            img = (smile_test[i][0]*1.0 - 128) / 255.0
            label = (int)(smile_test[i][1])
            test_data.append((img, one_hot(label, 5), 0.0))
        ###gender data###
        for i in range(len(gender_test)):
            img = (gender_test[i][0]*1.0 - 128) / 255.0
            label = (int)(gender_test[i][1])
            test_data.append((img, one_hot(label, 5), 1.0))
        for i in range(len(afadgender_test)):
            img = (afadgender_test[i][0]*1.0 - 128) / 255.0
            label = (int)(afadgender_test[i][1])
            test_data.append((img, one_hot(label, 5), 1.0))
        ###glasses data###
        for i in range(len(glasses_test)):
            img = (glasses_test[i][0]*1.0 - 128) / 255.0
            label = (int)(glasses_test[i][1])
            test_data.append((img, one_hot(label, 5), 2.0))
        ###ethnic data###
        for i in range(len(ethnic_test)):
            img = (ethnic_test[i][0]*1.0 - 128) / 255.0
            label = (int)(ethnic_test[i][1])
            test_data.append((img, one_hot(label, 5), 3.0))
        ###age data###
        for i in range(len(age_test)):
            img = (age_test[i][0]*1.0 - 128) / 255.0
            label = (int)(age_test[i][1])
            test_data.append((img, one_hot(label, 5), 4.0))

        saver = tf.train.Saver(max_to_keep=1)

        if not os.path.isfile(SAVE_FOLDER+'model.ckpt.index'):
            print('Create new model')
            sess.run(tf.global_variables_initializer())
            print('OK')
        else:
            print('Restoring existed model')
            saver.restore(sess, SAVE_FOLDER+'model.ckpt')
            print('OK')

        loss_summary_placeholder = tf.placeholder(tf.float32)

        smile_acc_summary_placeholder = tf.placeholder(tf.float32)
        gender_acc_summary_placeholder = tf.placeholder(tf.float32)
        glasses_acc_summary_placeholder = tf.placeholder(tf.float32)
        ethnic_acc_summary_placeholder = tf.placeholder(tf.float32)
        age_acc_summary_placeholder = tf.placeholder(tf.float32)

        smile_test_acc_summary_placeholder = tf.placeholder(tf.float32)
        gender_test_acc_summary_placeholder = tf.placeholder(tf.float32)
        glasses_test_acc_summary_placeholder = tf.placeholder(tf.float32)
        ethnic_test_acc_summary_placeholder = tf.placeholder(tf.float32)
        age_test_acc_summary_placeholder = tf.placeholder(tf.float32)
        
        tf.summary.scalar('loss', loss_summary_placeholder)

        tf.summary.scalar('smile acc', smile_acc_summary_placeholder)
        tf.summary.scalar('gender acc', gender_acc_summary_placeholder)
        tf.summary.scalar('glasses acc', glasses_acc_summary_placeholder)
        tf.summary.scalar('ethnic acc', ethnic_acc_summary_placeholder)
        tf.summary.scalar('age acc', age_acc_summary_placeholder)

        tf.summary.scalar('smile test acc', smile_test_acc_summary_placeholder)
        tf.summary.scalar('gender test acc', gender_test_acc_summary_placeholder)
        tf.summary.scalar('glasses test acc', glasses_test_acc_summary_placeholder)
        tf.summary.scalar('ethnic test acc', ethnic_test_acc_summary_placeholder)
        tf.summary.scalar('age test acc', age_test_acc_summary_placeholder)

        merge_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./summary/summary1/", graph=tf.get_default_graph())

        learning_rate = tf.get_collection('learning_rate')[0]
        current_epoch = (int)(global_step.eval(session=sess) / (len(train_data) // BATCH_SIZE))
        hist_loss = []
        min_loss = 1000
        
        for epoch in range(current_epoch, NUM_EPOCHS):
            
            print('Epoch:', str(epoch))
            np.random.shuffle(train_data)
            train_img = []
            train_label = []
            train_mask = []

            for i in range(len(train_data)):
                train_img.append(train_data[i][0])
                train_label.append(train_data[i][1])
                train_mask.append(train_data[i][2])

            number_batch = len(train_data) // BATCH_SIZE

            avg_ttl = []
            avg_rgl = []
            avg_smile_loss = []
            avg_gender_loss = []
            avg_glasses_loss = []
            avg_ethnic_loss = []
            avg_age_loss = []

            smile_nb_true_pred = 0
            gender_nb_true_pred = 0
            glasses_nb_true_pred = 0
            ethnic_nb_true_pred = 0
            age_nb_true_pred = 0

            smile_nb_train = 0
            gender_nb_train = 0
            glasses_nb_train = 0
            ethnic_nb_train = 0
            age_nb_train = 0

            print("Learning rate: %s" % str(learning_rate.eval(session=sess)))
            ####################training _color#####################
            for batch in range(number_batch):
                top = batch * BATCH_SIZE
                bot = min((batch + 1) * BATCH_SIZE, len(train_data))
                batch_img = np.asarray(train_img[top:bot])
                batch_label = np.asarray(train_label[top:bot])
                batch_mask = np.asarray(train_mask[top:bot])
                for i in range(BATCH_SIZE):
                    if batch_mask[i] == 0.0:
                        smile_nb_train += 1
                    elif batch_mask[i] == 1.0:
                        gender_nb_train += 1
                    elif batch_mask[i] == 2.0:
                        glasses_nb_train += 1
                    elif batch_mask[i] == 3.0:
                        ethnic_nb_train += 1
                    else:
                        age_nb_train += 1
                batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 3))
                batch_img = CNN2Head_input.augmentation(batch_img, 48)
                ttl, sml, gel, gll, etl, agl, l2l, _ = sess.run([loss, smile_loss, gender_loss, glasses_loss, ethnic_loss, age_loss,l2_loss, train_step],
                                                          feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                     phase_train: True,
                                                                     keep_prob: 0.5})
               
                print('Epoch:'+str(epoch)+' step %d'%batch+ ' total loss:'+ str(ttl) + '  smile loss: '+str(sml)+'  gender loss:'+str(gel) + '  glasses loss: '+str(gll) + ' ethnic loss: '+str(etl) + ' age loss: '+str(agl))

                smile_t, gender_t, glasses_t, ethnic_t, age_t = sess.run([smile_true_pred, gender_true_pred,glasses_true_pred,ethnic_true_pred,age_true_pred],
                                                                feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                phase_train: True,
                                                                keep_prob: 0.5})
                smile_nb_true_pred += smile_t
                gender_nb_true_pred += gender_t
                glasses_nb_true_pred += glasses_t
                ethnic_nb_true_pred += ethnic_t
                age_nb_true_pred += age_t

                avg_ttl.append(ttl)
                avg_smile_loss.append(sml)
                avg_gender_loss.append(gel)
                avg_glasses_loss.append(gll)
                avg_ethnic_loss.append(etl)
                avg_age_loss.append(agl)
                avg_rgl.append(l2l)
            #############training end###################

            #############test in the end of this epoch#################
            np.random.shuffle(test_data)
            test_img = []
            test_label = []
            test_mask = []
            epoch_loss = []

            for i in range(len(test_data)):
                test_img.append(test_data[i][0])
                test_label.append(test_data[i][1])
                test_mask.append(test_data[i][2])
            test_batch = len(test_data)//BATCH_SIZE
            
            smile_nb_true_pred_test = 0
            gender_nb_true_pred_test = 0
            glasses_nb_true_pred_test = 0
            ethnic_nb_true_pred_test = 0
            age_nb_true_pred_test = 0

            smile_nb_test = 0
            gender_nb_test = 0
            glasses_nb_test = 0
            ethnic_nb_test = 0
            age_nb_test = 0

            for batch in range(test_batch):
                top = batch * BATCH_SIZE
                bot = min((batch + 1) * BATCH_SIZE, len(test_data))
                batch_img = np.asarray(test_img[top:bot])
                batch_label = np.asarray(test_label[top:bot])
                batch_mask = np.asarray(test_mask[top:bot])
                for i in range(BATCH_SIZE):
                    if batch_mask[i] == 0.0:
                        smile_nb_test += 1
                    elif batch_mask[i] == 1.0:
                        gender_nb_test += 1
                    elif batch_mask[i] == 2.0:
                        glasses_nb_test += 1
                    elif batch_mask[i] == 3.0:
                        ethnic_nb_test += 1
                    else:
                        age_nb_test += 1
                batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 3))
                #batch_img = CNN2Head_input.augmentation(batch_img, 48)
                ttl, sml, gel, gll, etl, agl, l2l = sess.run([loss, smile_loss, gender_loss, glasses_loss, ethnic_loss, age_loss, l2_loss],
                                                        feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                    phase_train: False,
                                                                    keep_prob: 1.0})
                smile_t, gender_t, glasses_t, ethnic_t ,age_t = sess.run([smile_true_pred,gender_true_pred,glasses_true_pred,ethnic_true_pred,age_true_pred], feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                            phase_train: False,
                                                                            keep_prob: 1.0})
                smile_nb_true_pred_test += smile_t
                gender_nb_true_pred_test += gender_t
                glasses_nb_true_pred_test += glasses_t
                ethnic_nb_true_pred_test += ethnic_t
                age_nb_true_pred_test += age_t
                epoch_loss.append(ttl)

            epoch_loss = np.average(epoch_loss)
            # early stopping
            hist_loss.append(epoch_loss)
            patience = 20
            if hist_loss[epoch] < min_loss:
                patience_cnt = 0 #current lowest point
                min_loss = hist_loss[epoch]
                saver.save(sess, SAVE_FOLDER + 'model.ckpt')
            else:
                patience_cnt += 1
            if patience_cnt > patience:  # can't get a lower validation loss model in $patience epochs 
                print('early stopping at epoch %d...' % epoch)
                break

            # min_delta = 0.0001
            # if epoch > 0 and hist_loss[epoch-1] - hist_loss[epoch] > min_delta:
            #     patience_cnt = 0
            #     saver.save(sess, SAVE_FOLDER + 'model.ckpt')
            # else:
            #     patience_cnt += 1
            # if patience_cnt > patience:
            #     print("early stopping...")
            #     break
            smile_train_accuracy = smile_nb_true_pred * 1.0 / smile_nb_train
            gender_train_accuracy = gender_nb_true_pred * 1.0 / gender_nb_train
            glasses_train_accuracy = glasses_nb_true_pred * 1.0 / glasses_nb_train
            ethnic_train_accuracy = ethnic_nb_true_pred * 1.0 / ethnic_nb_train
            age_train_accuracy = age_nb_true_pred * 1.0 / age_nb_train

            smile_test_accuracy = smile_nb_true_pred_test * 1.0 / smile_nb_test
            gender_test_accuracy = gender_nb_true_pred_test * 1.0 / gender_nb_test
            glasses_test_accuracy = glasses_nb_true_pred_test * 1.0 / glasses_nb_test
            ethnic_test_accuracy = ethnic_nb_true_pred_test * 1.0 / ethnic_nb_test
            age_test_accuracy = age_nb_true_pred_test * 1.0 / age_nb_test

            avg_smile_loss = np.average(avg_smile_loss)
            avg_gender_loss = np.average(avg_gender_loss)
            avg_glasses_loss = np.average(avg_glasses_loss)
            avg_ethnic_loss = np.average(avg_ethnic_loss)
            avg_age_loss = np.average(avg_age_loss)
            avg_rgl = np.average(avg_rgl)
            avg_ttl = np.average(avg_ttl)

            
            summary = sess.run(merge_summary, feed_dict={loss_summary_placeholder: avg_ttl,
                                                         smile_acc_summary_placeholder:smile_train_accuracy,
                                                         gender_acc_summary_placeholder:gender_train_accuracy,
                                                         glasses_acc_summary_placeholder:glasses_train_accuracy,
                                                         ethnic_acc_summary_placeholder:ethnic_train_accuracy,
                                                         age_acc_summary_placeholder:age_train_accuracy,
                                                         smile_test_acc_summary_placeholder:smile_test_accuracy,
                                                         gender_test_acc_summary_placeholder:gender_test_accuracy,
                                                         glasses_test_acc_summary_placeholder:glasses_test_accuracy,
                                                         ethnic_test_acc_summary_placeholder:ethnic_test_accuracy,
                                                         age_test_acc_summary_placeholder:age_test_accuracy
                                                         })
            writer.add_summary(summary, global_step=epoch)

            print('Smile task train accuracy: ' + str(smile_train_accuracy * 100)+'%')
            print('Gender task train accuracy: ' + str(gender_train_accuracy * 100)+'%')
            print('Glasses task train accuracy: ' + str(glasses_train_accuracy * 100)+'%')
            print('Ethnic task train accuracy: ' + str(ethnic_train_accuracy * 100)+'%')
            print('Age task train accuracy: ' + str(age_train_accuracy * 100)+'%')
            print('\n')
            print('Smile task test accuracy: ' + str(smile_test_accuracy * 100)+'%')
            print('Gender task test accuracy: ' + str(gender_test_accuracy * 100)+'%')
            print('Glasses task test accuracy: ' + str(glasses_test_accuracy * 100)+'%')
            print('Ethnic task test accuracy: ' + str(ethnic_test_accuracy * 100)+'%')
            print('Age task test accuracy: ' + str(age_test_accuracy * 100)+'%')
            print('\n')
            print('Train Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
            print('Train Smile loss: ' + str(avg_smile_loss))
            print('Train Gender loss: ' + str(avg_gender_loss))
            print('Train Glasses loss: ' + str(avg_glasses_loss))
            print('Train Ethnic loss: ' + str(avg_ethnic_loss))
            print('Train Age loss: ' + str(avg_age_loss))
            print('\n')

            

if __name__ =='__main__':
    #训练微笑、眼镜、性别、人种、年龄段五个人脸属性模型
    train_multi_task()

