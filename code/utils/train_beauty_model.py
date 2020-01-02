import tensorflow as tf
import net
import CNNhead_input as CNN2Head_input
import numpy as np
from const import *
import os
import random

#[1,2,3,4,5] label 1point stored in idx0 of an array
def one_hot(index, num_classes):
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index-1]=1.0
    return tmp

def train_beauty_task():
    beauty_train, beauty_test = CNN2Head_input.getBeautyImage()
    with tf.Session() as sess:
        global_step = tf.contrib.framework.get_or_create_global_step()
        x,y_ = net.Input_beauty()
        y_beauty_conv ,phase_train, keep_prob = net.BeautyNETModel(x)
        beauty_loss , l2_loss , loss = net.beauty_loss(y_beauty_conv,y_)

        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

        train_step = net.train_op(loss, global_step)
        y_beauty = tf.get_collection('y_beauty')[0]   #tensor
        
        y_beauty_expectation = tf.cast(tf.argmax(y_beauty_conv,1)+1,dtype=tf.float32)
        y_beauty_ = tf.cast(tf.argmax(y_beauty,1)+1,dtype=tf.float32) #real score vector
        beauty_mae = tf.reduce_mean(tf.abs(tf.subtract(y_beauty_,y_beauty_expectation)))

        score_correct_prediction = tf.equal(tf.argmax(y_beauty_conv, 1), tf.argmax(y_beauty, 1))

        acc = tf.reduce_sum(tf.cast(score_correct_prediction, dtype=tf.float32)) / BATCH_SIZE
        train_data = []

        for i in range(len(beauty_train)):
            img = (beauty_train[i][0] * 1.0 - 128) / 255.0
            label = (int)(beauty_train[i][1])
            train_data.append((img,one_hot(label,5)))
    
        test_data = []
        for i in range(len(beauty_test)):
            img = (beauty_test[i][0] * 1.0 - 128) / 255.0
            label = (int)(beauty_test[i][1])
            test_data.append((img,one_hot(label,5)))

        print('-------------------------------------------------')
        print('whole trainning set:'+str(len(train_data)))
        print('whole testing set:'+str(len(test_data)))
        print('-------------------------------------------------')

        saver = tf.train.Saver(max_to_keep = 1)

        if not os.path.isfile(SAVE_FOLDER3+'model_beauty.ckpt.index'):
            print('Create new model')
            sess.run(tf.global_variables_initializer())
            print('OK')
        else:
            print('Restoring existed model')
            saver.restore(sess, SAVE_FOLDER3+'model_beauty.ckpt')
            print('OK')

        train_loss_summary_placeholder = tf.placeholder(tf.float32)
        test_loss_summary_placeholder = tf.placeholder(tf.float32)
        train_mae_summary_placeholder=tf.placeholder(tf.float32)
        test_mae_summary_placeholder=tf.placeholder(tf.float32)

        

        tf.summary.scalar('train_mae',train_mae_summary_placeholder)
        tf.summary.scalar('test_mae',test_mae_summary_placeholder)
        tf.summary.scalar('train_loss', train_loss_summary_placeholder)
        tf.summary.scalar('test_loss', test_loss_summary_placeholder)

        merge_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./summary/summary3/", graph=tf.get_default_graph())

        learning_rate = tf.get_collection('learning_rate')[0]
        current_epoch = (int)(global_step.eval(session = sess)/(len(train_data)//BATCH_SIZE))

        #######   training code  #######
        for epoch in range(current_epoch,NUM_EPOCHS):
            print('Epoch:', str(epoch))
            np.random.shuffle(train_data)
            train_img = []
            train_label = []

            for i in range(len(train_data)):
                train_img.append(train_data[i][0])
                train_label.append(train_data[i][1])
            
            number_batch = len(train_data)//BATCH_SIZE
            avg_ttl =[]  #total loss=beauty loss+l2 loss
            avg_rgl =[]  #l2loss
            avg_beauty_loss =[]  #beauty loss
            avg_mae = []  #mae per epoch
            avg_acc = []

            print("Learning rate: %s" % str(learning_rate.eval(session=sess)))
            for batch in range(number_batch):
                top = batch * BATCH_SIZE
                bot = min((batch + 1) * BATCH_SIZE, len(train_data))
                batch_img = np.asarray(train_img[top:bot])
                batch_label = np.asarray(train_label[top:bot])

                batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 3))
                batch_img = CNN2Head_input.augmentation(batch_img, 48)
                ttl ,l2l,bel,_ = sess.run([loss,l2_loss,beauty_loss,train_step],
                                          feed_dict={x:batch_img,y_:batch_label,phase_train:True,keep_prob:0.5})
                beauty_train_mae = sess.run(beauty_mae,
                                        feed_dict={x: batch_img, y_: batch_label, phase_train: True, keep_prob: 0.5})

                score_acc = sess.run(acc,
                                     feed_dict={x: batch_img, y_: batch_label, phase_train: True, keep_prob: 0.5})
                print('Epoch:'+str(epoch)+' step %d'%batch+' total loss:'+str(ttl)+' beauty_loss:'+str(bel) + ' mae of this batch:  '+str(beauty_train_mae) + ' acc of this batch:  '+str(score_acc*100)+'%')

                avg_beauty_loss.append(bel)   #per batch
                avg_ttl.append(ttl)
                avg_rgl.append(l2l)
                avg_mae.append(beauty_train_mae)
                avg_acc.append(score_acc)

            ######  testing onece in the end per epoch  #######
            np.random.shuffle(test_data)
            test_img = []
            test_label = []

            for i in range(len(test_data)):
                test_img.append(test_data[i][0])
                test_label.append(test_data[i][1])
            test_batch=len(test_data)//BATCH_SIZE
            avg_ttl_ =[]  #total loss=score loss+l2 loss in testing
            avg_rgl_ =[]  #l2loss in testing
            avg_beauty_loss_ =[]  #score loss in testing
            avg_mae_ = []  #mae in testing
            avg_acc_ = []


            for batch in range(test_batch):
                top = batch * BATCH_SIZE
                bot = min((batch + 1) * BATCH_SIZE, len(test_data))
                batch_img = np.asarray(test_img[top:bot])
                batch_label = np.asarray(test_label[top:bot])

                batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 3))
                # batch_img = CNN2Head_input.augmentation(batch_img, 48)
                ttl ,l2l,bel= sess.run([loss,l2_loss,beauty_loss],
                                          feed_dict={x:batch_img,y_:batch_label,phase_train:False,keep_prob:1.0})
                beauty_test_mae = sess.run(beauty_mae,
                                        feed_dict={x: batch_img, y_: batch_label, phase_train: False, keep_prob:1.0})
                score_acc_ = sess.run(acc,
                                     feed_dict={x: batch_img, y_: batch_label, phase_train: False, keep_prob: 1.0})

                avg_beauty_loss_.append(bel)   #per batch
                avg_ttl_.append(ttl)
                avg_rgl_.append(l2l)
                avg_mae_.append(beauty_test_mae)
                avg_acc_.append(score_acc_)


            # training loss and mae per epoch   
            avg_ttl = np.average(avg_ttl)  
            avg_rgl = np.average(avg_rgl)
            avg_beauty_loss = np.average(avg_beauty_loss)
            avg_mae = np.average(avg_mae)
            avg_acc = np.average(avg_acc)

            # testing loss and mae per epoch
            avg_ttl_ = np.average(avg_ttl_) 
            avg_rgl_ = np.average(avg_rgl_)
            avg_beauty_loss_ = np.average(avg_beauty_loss_)
            avg_mae_ = np.average(avg_mae_)
            avg_acc_ = np.average(avg_acc_)
    
            

            print('Final Beauty Train Mae'+ ' of Epoch'+str(epoch) +':  '+ str(avg_mae))
            print('Final Beauty Train Acc'+ ' of Epoch'+str(epoch) +':  '+ str(avg_acc*100)+'%')
            print('Final Beauty Train Beauty Loss'+ ' of Epoch'+str(epoch) +':  '+ str(avg_beauty_loss))
            print('Final Beauty Train Total Loss'+ ' of Epoch'+str(epoch) +':  '+ str(avg_ttl)+' L2-Loss: ' +str(avg_rgl))
            print('\n')

            print('Final Beauty Test Mae'+ ' of Epoch'+str(epoch) +':  '+ str(avg_mae_))
            print('Final Beauty Test Mae'+ ' of Epoch'+str(epoch) +':  '+ str(avg_acc_*100)+'%')
            print('Final Beauty Test Beauty Loss'+ ' of Epoch'+str(epoch) +':  '+ str(avg_beauty_loss_))
            print('Final Beauty Test Total Loss'+ ' of Epoch'+str(epoch) +':  '+ str(avg_ttl_)+' L2-Loss: ' +str(avg_rgl_))
            print('\n')


            saver.save(sess,SAVE_FOLDER3+'model_beauty.ckpt')

            summary = sess.run(merge_summary, feed_dict={train_loss_summary_placeholder: avg_ttl,
                                                         test_loss_summary_placeholder:avg_ttl_,
                                                         train_mae_summary_placeholder:avg_mae,
                                                         test_mae_summary_placeholder:avg_mae_
                                                         })
            writer.add_summary(summary, global_step=epoch)
            print('Eval finished,saving tensorboard scalars and graphs')

if __name__ == '__main__':
     train_beauty_task()


