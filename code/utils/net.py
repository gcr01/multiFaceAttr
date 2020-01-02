import tensorflow as tf
import numpy as np
from const import *


def _conv(name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        # n = filter_size * filter_size * 3 * out_filters #3D filter

        #filter = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
        #                        initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT)) 
        
        # #用varience  
        filter = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        return tf.nn.conv2d(x, filter, [1, strides, strides, 1], 'SAME')


def _relu(x, leakiness=0.1):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def _FC(name, x, out_dim, keep_rate, activation='relu'):
    assert (activation == 'relu') or (activation == 'softmax') or (activation == 'linear')
    with tf.variable_scope(name):
        dim = x.get_shape().as_list()
        dim = np.prod(dim[1:])
        x = tf.reshape(x, [-1, dim])
        #W = tf.get_variable('DW', [x.get_shape()[1], out_dim],
        #                    initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT))

        W = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer())
        x = tf.nn.xw_plus_b(x, W, b)
        if activation == 'relu':
            x = _relu(x)
        else:
            if activation == 'softmax':
                x = tf.nn.softmax(x)

        if activation != 'relu':
            return x
        else:
            return tf.nn.dropout(x, keep_rate)


def _max_pool(x, filter, stride):
    return tf.nn.max_pool(x, [1, filter, filter, 1], [1, stride, stride, 1], 'SAME')


def batch_norm(x, n_out, phase_train=True, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed



def VGG_ConvBlock(name, x, in_filters, out_filters, repeat, strides, phase_train):
    with tf.variable_scope(name):
        for layer in range(repeat):
            scope_name = name + '_' + str(layer)
            x = _conv(scope_name, x, 3, in_filters, out_filters, strides)
            if USE_BN:
                x = batch_norm(x, out_filters, phase_train)
            x = _relu(x)

            in_filters = out_filters

        x = _max_pool(x, 2, 2)
        return x


def Input():
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3]) # Placeholder:0
    y_ = tf.placeholder(tf.float32, [None, 5]) # Placeholder_1:0
    mask = tf.placeholder(tf.float32, [BATCH_SIZE]) # Placeholder_2:0

    return x, y_, mask

def Input_beauty():
    x = tf.placeholder(tf.float32, [None,IMG_SIZE,IMG_SIZE,3])
    y_ = tf.placeholder(tf.float32, [None,5])
    return x,y_

def BKNetModel(x):
    phase_train = tf.placeholder(tf.bool) # Placeholder_3:0
    keep_prob = tf.placeholder(tf.float32) # Placeholder_4:0

    x = VGG_ConvBlock('Block1', x, 3, 32, 2, 1, phase_train) 
    # print(x.get_shape())

    x = VGG_ConvBlock('Block2', x, 32, 64, 2, 1, phase_train)
    # print(x.get_shape())

    x = VGG_ConvBlock('Block3', x, 64, 128, 2, 1, phase_train)
    # print(x.get_shape())

    x = VGG_ConvBlock('Block4', x, 128, 256, 3, 1, phase_train)
    # print(x.get_shape())

    # Smile branch
    smile_fc1 = _FC('smile_fc1', x, 256, keep_prob)
    smile_fc2 = _FC('smile_fc2', smile_fc1, 256, keep_prob)
    y_smile_conv = _FC('smile_softmax', smile_fc2, 2, keep_prob, 'softmax')

    # Gender branch
    gender_fc1 = _FC('gender_fc1', x, 256, keep_prob)
    gender_fc2 = _FC('gender_fc2', gender_fc1, 256, keep_prob)
    y_gender_conv = _FC('gender_softmax', gender_fc2, 2, keep_prob, 'softmax')

    # Glasses branch
    glasses_fc1 = _FC('glasses_fc1', x, 256, keep_prob)
    glasses_fc2 = _FC('glasses_fc2', glasses_fc1, 256, keep_prob)
    y_glasses_conv = _FC('glasses_softmax', glasses_fc2, 2, keep_prob, 'softmax')

    # Ethnic branch,added at OCt.10.16
    ethnic_fc1 = _FC('ethnic_fc1', x, 256, keep_prob)
    ethnic_fc2 = _FC('ethnic_fc2', ethnic_fc1, 256, keep_prob)
    y_ethnic_conv = _FC('ethnic_softmax', ethnic_fc2, 5, keep_prob, 'softmax')

    # Age branch
    age_fc1 = _FC('age_fc1',x,256,keep_prob)
    age_fc2 = _FC('age_fc2',age_fc1,256,keep_prob)
    y_age_conv = _FC('age_softmax',age_fc2,5,keep_prob,'softmax')
    
    #return y_smile_conv, y_gender_conv, y_glasses_conv,phase_train, keep_prob
    return y_smile_conv, y_gender_conv, y_glasses_conv, y_ethnic_conv, y_age_conv, phase_train, keep_prob

def BeautyNETModel(x):
    phase_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    x = VGG_ConvBlock('Block1', x, 3, 32, 2, 1, phase_train)
    # print(x.get_shape())

    x = VGG_ConvBlock('Block2', x, 32, 64, 2, 1, phase_train)
    # print(x.get_shape())

    x = VGG_ConvBlock('Block3', x, 64, 128, 2, 1, phase_train)
    # print(x.get_shape())

    x = VGG_ConvBlock('Block4', x, 128, 256, 3, 1, phase_train)
    # print(x.get_shape())

    #Beauty branch
    beauty_fc1 = _FC('beauty_fc1',x,256,keep_prob)
    beauty_fc2 = _FC('beauty_fc2',beauty_fc1,256,keep_prob)
    y_beauty_conv = _FC('beauty_softmax',beauty_fc2,5,keep_prob,'softmax')

    return y_beauty_conv, phase_train, keep_prob


def beauty_loss(y_beauty_conv,y_):
    safe_log = tf.clip_by_value(y_beauty_conv, 1e-10, 1e100)
    y_beauty = tf.slice(y_,[0,0],[BATCH_SIZE,5])
    tf.add_to_collection('y_beauty',y_beauty)
    beauty_cross_entropy = tf.reduce_sum(
         tf.reduce_sum(-y_beauty * tf.log(safe_log), axis=1) / BATCH_SIZE)
    l2_loss = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
            l2_loss.append(tf.nn.l2_loss(var))
    l2_loss = WEIGHT_DECAY * tf.add_n(l2_loss)
    total_loss = beauty_cross_entropy + l2_loss
    return beauty_cross_entropy ,l2_loss ,total_loss

def selective_loss(y_smile_conv, y_gender_conv, y_glasses_conv,y_ethnic_conv, y_age_conv, y_, mask):
    
    # use function to make the follwing code short and clean
    safe_smilelog = tf.clip_by_value(y_smile_conv, 1e-10, 1e100)
    safe_genderlog = tf.clip_by_value(y_gender_conv, 1e-10, 1e100)
    safe_glasseslog = tf.clip_by_value(y_glasses_conv, 1e-10, 1e100)
    safe_ethniclog = tf.clip_by_value(y_ethnic_conv, 1e-10, 1e100)
    safe_agelog = tf.clip_by_value(y_age_conv, 1e-10, 1e100)

    vector_zero = tf.constant(0., tf.float32, [BATCH_SIZE]) #a vector with all zeros,used for comparing with mask and get the No. of smile data this batch  
    vector_one = tf.constant(1., tf.float32, [BATCH_SIZE])  #for gender
    vector_two = tf.constant(2., tf.float32, [BATCH_SIZE])  #for glass
    vector_three = tf.constant(3., tf.float32, [BATCH_SIZE])  #for ethnic
    vector_four = tf.constant(4., tf.float32, [BATCH_SIZE]) #for age

    smile_mask = tf.cast(tf.equal(mask, vector_zero), tf.float32)
    gender_mask = tf.cast(tf.equal(mask, vector_one), tf.float32)
    glasses_mask = tf.cast(tf.equal(mask, vector_two), tf.float32)
    ethnic_mask = tf.cast(tf.equal(mask, vector_three), tf.float32)
    age_mask = tf.cast(tf.equal(mask, vector_four), tf.float32)

    tf.add_to_collection('smile_mask', smile_mask)
    tf.add_to_collection('gender_mask', gender_mask)
    tf.add_to_collection('glasses_mask', glasses_mask)
    tf.add_to_collection('ethnic_mask', ethnic_mask)
    tf.add_to_collection('age_mask', age_mask)

    y_smile = tf.slice(y_, [0, 0], [BATCH_SIZE, 2])  #cut the y_'s size from 5 to 2
    y_gender = tf.slice(y_, [0, 0], [BATCH_SIZE, 2])
    y_glasses = tf.slice(y_, [0, 0], [BATCH_SIZE, 2])
    y_ethnic = tf.slice(y_, [0, 0], [BATCH_SIZE, 5])
    y_age = tf.slice(y_, [0, 0], [BATCH_SIZE, 5])

    tf.add_to_collection('y_smile', y_smile)
    tf.add_to_collection('y_gender', y_gender)
    tf.add_to_collection('y_glasses', y_glasses)
    tf.add_to_collection('y_ethnic', y_ethnic)
    tf.add_to_collection('y_age', y_age)

    # smile_cross_entropy = tf.reduce_sum(
    #     tf.reduce_sum(-y_smile * tf.log(safe_smilelog), axis=1) * smile_mask) / tf.clip_by_value(
    #     tf.reduce_sum(smile_mask), 1, 1e9)
    # gender_cross_entropy = tf.reduce_sum(
    #     tf.reduce_sum(-y_gender * tf.log(safe_genderlog), axis=1) * gender_mask) / tf.clip_by_value(
    #     tf.reduce_sum(gender_mask), 1, 1e9)
    # glasses_cross_entropy = tf.reduce_sum(
    #     tf.reduce_sum(-y_glasses * tf.log(safe_glasseslog), axis=1) * glasses_mask) / tf.clip_by_value(
    #     tf.reduce_sum(glasses_mask), 1, 1e9)
    # ethnic_cross_entropy = tf.reduce_sum(
    #     tf.reduce_sum(-y_ethnic * tf.log(safe_ethniclog), axis=1) * ethnic_mask) / tf.clip_by_value(
    #     tf.reduce_sum(ethnic_mask), 1, 1e9)
    # age_cross_entropy = tf.reduce_sum(
    #     tf.reduce_sum(-y_age * tf.log(safe_agelog), axis=1) * age_mask) / tf.clip_by_value(
    #     tf.reduce_sum(age_mask), 1, 1e9)
    
    #11.28 Gucongrong
    #left: (1-y_)^gamma ,right alpha*log(y_), focal : left*right = alpha*(1-y_)^gamma*log(y_)

    # use function to make the follwing code short and clean
    gamma = 2
    weight_smileset = [0.7,1]  #according to the weight of each class in dataset 
    multi = tf.reduce_sum(y_smile * safe_smilelog, axis = 1)  #[p0,p1,p2,...]
    one_sub = tf.ones_like(multi)  #[1,1,1,...]
    focal_smileleft = tf.pow( (one_sub-multi), gamma )*smile_mask   #([(1-p0)^gamma,(1-p1)^gamma,(1-p2)^gamma,...] positive
    focal_smileright = tf.reduce_sum(tf.log(safe_smilelog)*weight_smileset*y_smile, axis = 1)*smile_mask  #[alpha0*logp0, alpha1*logp1, ...] negtive
    smile_focalloss = -tf.reduce_sum(focal_smileleft * focal_smileright) / tf.clip_by_value(
        tf.reduce_sum(smile_mask), 1, 1e9)

    weight_genderset = [1,1]  
    multi = tf.reduce_sum(y_gender * safe_genderlog, axis = 1)  
    one_sub = tf.ones_like(multi) 
    focal_genderleft = tf.pow( (one_sub-multi), gamma )*gender_mask   
    focal_genderright = tf.reduce_sum(tf.log(safe_genderlog)*weight_genderset*y_gender, axis = 1)*gender_mask  
    gender_focalloss = -tf.reduce_sum(focal_genderleft * focal_genderright) / tf.clip_by_value(
        tf.reduce_sum(gender_mask), 1, 1e9)

    weight_glassesset = [0.7,1]  
    multi = tf.reduce_sum(y_glasses * safe_glasseslog, axis = 1)  
    one_sub = tf.ones_like(multi)  
    focal_glassesleft = tf.pow( (one_sub-multi), gamma )*glasses_mask  
    focal_glassesright = tf.reduce_sum(tf.log(safe_glasseslog)*weight_glassesset*y_glasses, axis = 1)*glasses_mask  
    glasses_focalloss = -tf.reduce_sum(focal_glassesleft * focal_glassesright) / tf.clip_by_value(
        tf.reduce_sum(glasses_mask), 1, 1e9)

    
    weight_ethnicset = [1, 1, 1, 1, 1]  
    multi = tf.reduce_sum(y_ethnic * safe_ethniclog, axis = 1)  
    
    focal_ethnicleft = tf.pow( (one_sub-multi), gamma )*ethnic_mask  
    focal_ethnicright = tf.reduce_sum(tf.log(safe_ethniclog)*weight_ethnicset*y_ethnic, axis = 1)*ethnic_mask  
    ethnic_focalloss = -tf.reduce_sum(focal_ethnicleft * focal_ethnicright) / tf.clip_by_value(
        tf.reduce_sum(ethnic_mask), 1, 1e9)
    
    weight_ageset = [0.85, 0.9, 0.4, 0.8, 1.0]  
    multi = tf.reduce_sum(y_age * safe_agelog, axis = 1)  
    one_sub = tf.ones_like(multi)  
    focal_ageleft = tf.pow( (one_sub-multi), gamma )*age_mask   
    focal_ageright = tf.reduce_sum(tf.log(safe_agelog)*weight_ageset*y_age, axis = 1)*age_mask  
    age_focalloss = -tf.reduce_sum(focal_ageleft * focal_ageright) / tf.clip_by_value(
        tf.reduce_sum(age_mask), 1, 1e9)


    l2_loss = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
            l2_loss.append(tf.nn.l2_loss(var))
    l2_loss = WEIGHT_DECAY * tf.add_n(l2_loss)

    # total_loss = 1*smile_cross_entropy + 1*gender_cross_entropy + 1*glasses_cross_entropy + 1*ethnic_cross_entropy + 1*age_cross_entropy + 1*l2_loss
    
    total_loss = 1*smile_focalloss + 1*gender_focalloss + 1*glasses_focalloss + ethnic_focalloss + age_focalloss + 1*l2_loss
    print('all focal loss')

    # return smile_cross_entropy, gender_cross_entropy, glasses_cross_entropy, ethnic_cross_entropy, age_cross_entropy, l2_loss, total_loss
    return smile_focalloss, gender_focalloss, glasses_focalloss, ethnic_focalloss, age_focalloss, l2_loss, total_loss



def train_op(loss, global_step):

    #每训完一个batch，globalstep加1，DECAY_STEP=2000,意味着每2000个batch学习率衰减一次，
    #每个epoch295个batch，也就是大约每7个epoch对学习率做一次衰减2000个epoch一共衰减了约300次
    #约第1300个epoch时衰减为1e-6，实验证明虽然有adamoptimizer，但初始学习率较大时使用学习率衰减有一定效果

    learning_rate = tf.train.exponential_decay(INIT_LR, global_step, DECAY_STEP, DECAY_LR_RATE, staircase=True)
    ##train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss,global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    tf.add_to_collection('learning_rate', learning_rate)
    return train_step
