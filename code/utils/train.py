import CNNhead_input as CNN2Head_input
import os
import tensorflow as tf
import numpy as np
import net
from const import *
import random
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--train_model',required=True,
                help = 'Choose train function')
args = vars(ap.parse_args())



def one_hot(index, num_classes):

    tmp = np.zeros(num_classes, dtype=np.float32)
    if(index==-1):
        tmp[0] = 1.0
    elif(index==1):
        tmp[1]=1.0
    else:
        tmp[index] = 1.0
    return tmp

def train_multi_task():
    ''' PREPARE DATA '''
    smile_train, smile_test = CNN2Head_input.getSmileImage()
    gender_train, gender_test = CNN2Head_input.getGenderImage()
    glasses_train, glasses_test test= CNN2Head_input.getGlassesImage()
    with tf.Session() as sess:
        global_step = tf.contrib.framework.get_or_create_global_step()
        x, y_, mask = net.Input()

        y_smile_conv, y_gender_conv, y_glasses_conv, phase_train, keep_prob = net.BKNetModel(x)

        smile_loss, gender_loss, glasses_loss, l2_loss, loss = net.selective_loss(y_smile_conv, y_gender_conv,
                                                                                     y_glasses_conv, y_, mask)
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

        train_step = net.train_op(loss, global_step)

        smile_mask = tf.get_collection('smile_mask')[0]
        gender_mask = tf.get_collection('gender_mask')[0]
        glasses_mask = tf.get_collection('glasses_mask')[0]

        y_smile = tf.get_collection('y_smile')[0]
        y_gender = tf.get_collection('y_gender')[0]
        y_glasses = tf.get_collection('y_glasses')[0]

        smile_correct_prediction = tf.equal(tf.argmax(y_smile_conv, 1), tf.argmax(y_smile, 1))
        gender_correct_prediction = tf.equal(tf.argmax(y_gender_conv, 1), tf.argmax(y_gender, 1))
        glasses_correct_prediction = tf.equal(tf.argmax(y_glasses_conv, 1), tf.argmax(y_glasses, 1))

        smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32) * smile_mask)
        gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32) * gender_mask)
        glasses_true_pred = tf.reduce_sum(tf.cast(glasses_correct_prediction, dtype=tf.float32) * glasses_mask)
        # age_mae, update_op = tf.metrics.mean_absolute_error(
        #     tf.argmax(y_glasses, 1), tf.argmax(y_glasses_conv, 1), name="age_mae")


        train_data = []
        # Mask: Smile -> 0, Gender -> 1, Glasses -> 2
        for i in range(len(smile_train)):
            img = (smile_train[i][0] - 128) / 255.0
            label = (int)(smile_train[i][1])
            train_data.append((img, one_hot(label, 2), 0.0))
        for i in range(len(gender_train)):
            img = (gender_train[i][0] - 128) / 255.0
            label = (int)(gender_train[i][1])
            train_data.append((img, one_hot(label, 2), 1.0))
        for i in range(len(glasses_train)):
            img = (glasses_train[i][0] - 128) / 255.0
            label = (int)(glasses_train[i][1])
            train_data.append((img, one_hot(label, 2), 2.0))

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
        tf.summary.scalar('loss', loss_summary_placeholder)
        merge_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./summary/summary1/", graph=tf.get_default_graph())

        learning_rate = tf.get_collection('learning_rate')[0]
        current_epoch = (int)(global_step.eval(session=sess) / (len(train_data) // BATCH_SIZE))
        for epoch in range(current_epoch + 1, NUM_EPOCHS):
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

            smile_nb_true_pred = 0
            gender_nb_true_pred = 0
            glasses_nb_true_pred = 0

            smile_nb_train = 0
            gender_nb_train = 0
            glasses_nb_train = 0

            print("Learning rate: %f" % learning_rate.eval(session=sess))
        #     for batch in range(number_batch):
            for batch in range(number_batch):
                top = batch * BATCH_SIZE
                bot = min((batch + 1) * BATCH_SIZE, len(train_data))
                batch_img = np.asarray(train_img[top:bot])
                batch_label = np.asarray(train_label[top:bot])
                batch_mask = np.asarray(train_mask[top:bot])
                for i in range(BATCH_SIZE):
                    if batch_mask[i] == 0.0:
                            smile_nb_train += 1
                    else:
                        if batch_mask[i] == 1.0:
                            gender_nb_train += 1
                        else:
                            glasses_nb_train += 1
                batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 1))
                batch_img = CNN2Head_input.augmentation(batch_img, 48)
                ttl, sml, gel, gll, l2l, _ = sess.run([loss, smile_loss, gender_loss, glasses_loss, l2_loss, train_step],
                                                          feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                     phase_train: True,
                                                                     keep_prob: 0.5})
                print('Epoch:'+str(epoch)+' step %d'%batch+ ' total loss:'+ str(ttl) + '  smile loss: '+str(sml)+'  gender loss:'+str(gel) + '  glasses loss: '+str(gll))

                smile_nb_true_pred += sess.run(smile_true_pred, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                               phase_train: False,
                                                                               keep_prob: 0.5})

                gender_nb_true_pred += sess.run(gender_true_pred,
                                                    feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                               phase_train: False,
                                                               keep_prob: 0.5})

                glasses_nb_true_pred += sess.run(glasses_true_pred,
                                                     feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                phase_train:False,
                                                                keep_prob: 0.5})
                avg_ttl.append(ttl)
                avg_smile_loss.append(sml)
                avg_gender_loss.append(gel)
                avg_glasses_loss.append(gll)

                avg_rgl.append(l2l)

            smile_train_accuracy = smile_nb_true_pred * 1.0 / smile_nb_train
            gender_train_accuracy = gender_nb_true_pred * 1.0 / gender_nb_train
            glasses_train_accuracy = glasses_nb_true_pred * 1.0 / glasses_nb_train

            avg_smile_loss = np.average(avg_smile_loss)
            avg_gender_loss = np.average(avg_gender_loss)
            avg_glasses_loss = np.average(avg_glasses_loss)

            avg_rgl = np.average(avg_rgl)
            avg_ttl = np.average(avg_ttl)

            summary = sess.run(merge_summary, feed_dict={loss_summary_placeholder: avg_ttl})
            writer.add_summary(summary, global_step=epoch)

            with open('log.csv', 'a+') as f:
                # epochs, smile_train_accuracy, gender_train_accuracy, glasses_train_accuracy,
                # avg_smile_loss, avg_gender_loss, avg_age_loss, avg_ttl, avg_rgl
                f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(epoch, smile_train_accuracy, gender_train_accuracy,
                                                                       glasses_train_accuracy, avg_smile_loss, avg_gender_loss,
                                                                       avg_glasses_loss, avg_ttl, avg_rgl))

            print('Smile task train accuracy: ' + str(smile_train_accuracy * 100))
            print('Gender task train accuracy: ' + str(gender_train_accuracy * 100))
            print('Glasses task train accuracy: ' + str(glasses_train_accuracy * 100))

            print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
            print('Smile loss: ' + str(avg_smile_loss))
            print('Gender loss: ' + str(avg_gender_loss))
            print('Glasses loss: ' + str(avg_glasses_loss))

            print('\n')

            saver.save(sess, SAVE_FOLDER + 'model.ckpt')
def train_ethnic_task():
    ''' PREPARE DATA '''
    ethnic_train, ethnic_test = CNN2Head_input.getEthnicImage()
    with tf.Session() as sess:
        global_step = tf.contrib.framework.get_or_create_global_step()
        x, y_ = net.Input_ethnic()
        y_ethnic_conv, phase_train, keep_prob = net.ETHNICModel(x)

        ethnic_loss, l2_loss, loss = net.ethnic_loss(y_ethnic_conv,y_)
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

        train_step = net.train_op(loss, global_step)

        y_ethnic = tf.get_collection('y_ethnic')[0]
        #ethnic_correct_prediction = tf.equal(tf.argmax(y_ethnic_conv, 1), tf.argmax(y_ethnic, 1))
        ethnic_correct_prediction = tf.equal(tf.argmax(y_ethnic_conv, 1), tf.argmax(y_, 1))
        ethnic_true_pred = tf.reduce_sum(tf.cast(ethnic_correct_prediction, dtype=tf.float32))
        train_data = []
        for i in range(len(ethnic_train)):
            img = (ethnic_train[i][0] - 128) / 255.0
            label = (int)(ethnic_train[i][1])
            train_data.append((img, one_hot(label, 5)))
        test_data =[]
        for i in range(len(ethnic_test)):
            img = (ethnic_test[i][0] - 128) / 255.0
            label = (int)(ethnic_test[i][1])
            test_data.append((img, one_hot(label, 5)))

        saver = tf.train.Saver(max_to_keep=1)

        if not os.path.isfile(SAVE_FOLDER3+'model-ethnic.ckpt.index'):
            print('Create new model')
            sess.run(tf.global_variables_initializer())
            print('OK')
        else:
            print('Restoring existed model')
            saver.restore(sess, SAVE_FOLDER3+'model-ethnic.ckpt')
            print('OK')

        loss_summary_placeholder = tf.placeholder(tf.float32)
        tf.summary.scalar('loss', loss_summary_placeholder)
        merge_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./summary/summary3/", graph=tf.get_default_graph())

        learning_rate = tf.get_collection('learning_rate')[0]
        current_epoch = (int)(global_step.eval(session=sess) / (len(train_data) // BATCH_SIZE))
        for epoch in range(current_epoch + 1, NUM_EPOCHS):
            print('Epoch:', str(epoch))
            np.random.shuffle(train_data)
            train_img = []
            train_label = []
            test_img=[]
            test_label=[]

            for i in range(len(train_data)):
                train_img.append(train_data[i][0])
                train_label.append(train_data[i][1])
            # for i in range(len(test_data)):
            #     test_img.append(test_data[i][0])
            #     test_label.append(test_data[i][1])
            # test_img = list(np.reshape(test_img,(-1,48,48,1)))
            # test_batch_img = random.sample(test_img,BATCH_SIZE)
            # test_batch_label = random.sample(test_label,BATCH_SIZE)

            number_batch = len(train_data) // BATCH_SIZE

            avg_ttl = []
            avg_rgl = []
            avg_ethnic_loss = []
            ethnic_nb_true_pred = 0
            print("Learning rate: %f" % learning_rate.eval(session=sess))
            for batch in range(number_batch):
                top = batch * BATCH_SIZE
                bot = min((batch + 1) * BATCH_SIZE, len(train_data))
                batch_img = np.asarray(train_img[top:bot])
                batch_label = np.asarray(train_label[top:bot])
                batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 1))
                batch_img = CNN2Head_input.augmentation(batch_img, 48)
                ttl, etl, l2l, _ = sess.run([loss, ethnic_loss, l2_loss, train_step],
                                                          feed_dict={x: batch_img, y_: batch_label,
                                                                     phase_train: True,
                                                                     keep_prob: 0.5})
                print('Epoch:' + str(epoch) + ' step %d' % batch + ' total loss:' + str(ttl) + '  ethnic loss: ' + str(
                    etl))

                ethnic_nb_true_pred += sess.run(ethnic_true_pred,
                                                feed_dict={x: batch_img, y_: batch_label,
                                                           phase_train: True,
                                                           keep_prob: 0.5})
                avg_ttl.append(ttl)
                avg_ethnic_loss.append(etl)
                avg_rgl.append(l2l)
            ethnic_train_accuracy = ethnic_nb_true_pred * 1.0 / len(train_data)
            # ethnic_nb_true_pred_test = sess.run(ethnic_true_pred,
            #                                    feed_dict={x:test_img , y_: test_label,
            #                                               phase_train: True,
            #                                               keep_prob: 0.5})
            # ethnic_test_accuracy = ethnic_nb_true_pred_test*1.0/len(test_img)

            avg_rgl = np.average(avg_rgl)
            avg_ttl = np.average(avg_ttl)
            avg_ethnic_loss = np.average(avg_ethnic_loss)
            summary = sess.run(merge_summary, feed_dict={loss_summary_placeholder: avg_ttl})
            writer.add_summary(summary, global_step=epoch)
            print('Ethnic task train accuracy: ' + str(ethnic_train_accuracy * 100))
            # print('Ethnic task test accuracy: '+str(ethnic_test_accuracy*100))
            print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
            print('Ethnic loss: ' + str(avg_ethnic_loss))

            print('\n')

            saver.save(sess, SAVE_FOLDER3 + 'model-ethnic.ckpt')



if __name__ =='__main__':
    if(int(args['train_model'])==1): #输入参数为1时训练微笑、眼镜、性别三属性模型
        train_multi_task()
    elif(int(args['train_model'])==3): #输入参数为3时训练人种模型
        train_ethnic_task()

