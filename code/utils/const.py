WEIGHT_INIT = 0.01
NUM_TASKS = 3
IMG_SIZE = 48
INIT_LR = 0.0001
USE_BN = True
BN_DECAY = 0.99
EPSILON = 0.0001
WEIGHT_DECAY = 0.01
DECAY_STEP = 3000
DECAY_LR_RATE = 0.95
BATCH_SIZE = 512

USE_GPU = True
SAVE_FOLDER = './save/current/' # 微笑、眼镜、性别多任务模型存放路径
SAVE_FOLDER2 = './save/current2/' # 单独年龄分类模型存放路径
SAVE_FOLDER3 = './save/current3/' # 单独人种分类模型存放路径
SAVE_FOLDER4 = './save/current4/' # 年龄、人种多任务模型存放路径

NUM_EPOCHS = 1500
DROP_OUT_PROB = 0.5
