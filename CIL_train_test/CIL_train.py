import os

from agents.imitation.imitation_learning_network import load_imitation_learning_network
import numpy as np
from data import DataManager
import CIL_loss
import tensorflow as tf
import matplotlib.pyplot as plt

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)


# -----------------超参数
WANTED_STEP = 5000
WANTED_TRUE = False
EPOCH_NUM = 160
BATCH_SIZE = 200
image_size = (88, 200, 3)
TARGET_NUM = 3
DROP_OUT_TRAIN = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
DROP_OUT_TEST = [1.0 for i in DROP_OUT_TRAIN]
BRANCH_WEI = [0.95, 0.95, 0.95, 0.95, 0.05]
VARIABLE_WEI = {'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05}
LEARNING_RATE = 1e-4
# THRESHOLD = 1000
# DECAY = 0.5
BATCH_NUM_PER_AGENT = None
GRAPH_LOG = False
global_step = tf.train.get_or_create_global_step()
test_loss_lowest = 100

# -------------数据集相关
train_database_path = r'/home/mist/AgentHuman/SeqTrain'
test_database_path = r'/home/mist/AgentHuman/SeqVal'
train_data_manager = DataManager(train_database_path, TARGET_NUM)
test_data_manager = DataManager(test_database_path, TARGET_NUM)

# ------------loss图像相关
train_loss_log = []
train_loss_epoch_log = []
test_loss_log = []
test_loss_epoch_log = []

# --------------创建静态图
# placeholder
image_input = tf.placeholder("float", shape=[None, image_size[0], image_size[1], image_size[2]], name="input_image")
speed_mea = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1], name="speed_mea")
target = tf.placeholder(tf.float32, shape=[None, TARGET_NUM], name="target")
input = [None, speed_mea]

_d_out = tf.placeholder(tf.float32, shape=[len(DROP_OUT_TRAIN)])
mask_b1 = tf.placeholder(tf.float32, shape=[None, TARGET_NUM], name='follow_lane')
mask_b2 = tf.placeholder(tf.float32, shape=[None, TARGET_NUM], name='left')
mask_b3 = tf.placeholder(tf.float32, shape=[None, TARGET_NUM], name='right')
mask_b4 = tf.placeholder(tf.float32, shape=[None, TARGET_NUM], name='straight')
mask = [mask_b1, mask_b2, mask_b3, mask_b4]

# Network, loss and optimizer
with tf.name_scope('Network'):
    branches = load_imitation_learning_network(image_input, input, image_size, _d_out)
loss = CIL_loss.loss_compute(mask, branches, target, speed_mea, BRANCH_WEI, VARIABLE_WEI, BATCH_SIZE)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
# temp = set(tf.global_variables())


# ----------------模型保存及loss记录相关
saver = tf.train.Saver(max_to_keep=10)
best_saver = tf.train.Saver(max_to_keep=3)
model_path = tf.train.latest_checkpoint(r'log4')
print(model_path)
writer = tf.summary.FileWriter(r'log4/loss/train_loss')
test_writer = tf.summary.FileWriter(r'log4/loss/test_loss')
loss_summary = tf.summary.scalar('loss', loss)

# --------------------------创建会话
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    # --------------模型加载
        if model_path:
            saver.restore(sess, model_path)
            print(f'Restore from {model_path}')
        else:
            sess.run(tf.global_variables_initializer())
            print('Initialize from the scratch')

        for epoch in range(EPOCH_NUM):
            if WANTED_TRUE:
                print('wanted achieve')
                break
            train_batch_path_list = train_data_manager.shuffle_data(BATCH_NUM_PER_AGENT)
            test_batch_path_list = test_data_manager.shuffle_data(BATCH_NUM_PER_AGENT)
            for batch_ctr in range(len(train_batch_path_list)):
                # ----------------准备数据
                train_batch_images_shuffled, train_batch_labels_shuffled, train_file_wrong = \
                    train_data_manager.get_shuffled_batch(train_batch_path_list[batch_ctr])
                test_batch_images_shuffled, test_batch_labels_shuffled, test_file_wrong = \
                    test_data_manager.get_shuffled_batch(test_batch_path_list[batch_ctr % len(test_batch_path_list)])
                if train_file_wrong or test_file_wrong:
                #if train_file_wrong:
                    continue

                train_mask_b1, train_mask_b2, train_mask_b3, train_mask_b4 = \
                    train_data_manager.command(train_batch_labels_shuffled)
                test_mask_b1, test_mask_b2, test_mask_b3, test_mask_b4 = \
                    test_data_manager.command(test_batch_labels_shuffled)

                train_target_in, train_speed_mea_in = train_data_manager.get_target_input(train_batch_labels_shuffled)
                test_target_in, test_speed_mea_in = test_data_manager.get_target_input(test_batch_labels_shuffled)

                # # -----------------------记录静态图
                # if GRAPH_LOG:
                #     GRAPH_LOG = False
                #     tf.summary.FileWriter('logs/', sess.graph)

                # -----------------------训练
                _, train_loss_batch, train_loss_summary = sess.run([train_step, loss, loss_summary],
                                                                   feed_dict={image_input: train_batch_images_shuffled,
                                                                              speed_mea: train_speed_mea_in,
                                                                              target: train_target_in,
                                                                              mask_b1: train_mask_b1,
                                                                              mask_b2: train_mask_b2,
                                                                              mask_b3: train_mask_b3,
                                                                              mask_b4: train_mask_b4,
                                                                              _d_out: DROP_OUT_TRAIN})

                test_loss_batch, test_loss_summary = sess.run([loss, loss_summary],
                                                              feed_dict={image_input: test_batch_images_shuffled,
                                                                         speed_mea: test_speed_mea_in,
                                                                         target: test_target_in,
                                                                         mask_b1: test_mask_b1,
                                                                         mask_b2: test_mask_b2,
                                                                         mask_b3: test_mask_b3,
                                                                         mask_b4: test_mask_b4,
                                                                        _d_out: DROP_OUT_TEST})
                # # b1, b2, b3, b4, speed = sess.run([branches[0], branches[1], branches[2], branches[3], branches[4]],
                                                 #                           feed_dict={image_input: test_batch_images_shuffled,
                                                 #                                      speed_mea: test_speed_mea_in,
                                                 #                                      target: test_target_in,
                                                 #                                      _d_out: DROP_OUT_TEST})
                #
                #
                # print(test_target_in[:1])
                # print()
                # print(b1[:1])
                # print()
                # print(b2[:1])
                # print()
                # print(b3[:1])
                # print()
                # print(b4[:1])
                # print()
                # print(speed[:1])
                # print(test_mask_b1[:1])
                # print(test_mask_b2[:1])
                # print(test_mask_b3[:1])
                # print(test_mask_b4[:1])

                # -----------------------数据记录与模型保存
                writer.add_summary(train_loss_summary, global_step=sess.run(global_step))
                test_writer.add_summary(test_loss_summary, global_step=sess.run(global_step))
                if global_step == WANTED_STEP:
                    saver.save(sess, 'log4/model', global_step=global_step, write_meta_graph=False)
                    WANTED_TRUE = True
                    break

                # ------------------------模型选择
                if test_loss_lowest > test_loss_batch:
                    test_loss_lowest = test_loss_batch
                    best_saver.save(sess, 'log4/model/best_model', global_step=global_step)

                # ------------------------数据记录
                train_loss_log.append(train_loss_batch)
                test_loss_log.append(test_loss_batch)
                print(f'EPOCH:{epoch+1}\tGLOBAL_STEP:{sess.run(global_step)}\tSTEP:{batch_ctr+1}\t'
                       f'Train_loss:{train_loss_batch}\tTest_loss:{test_loss_batch}\tTest_loss_lowest:{test_loss_lowest}')

# try:
#     fig, ax = plt.subplots()
#     x = np.arange(len(train_loss_log))
#     line1, = ax.plot(x, np.array(train_loss_log), label='train loss')
#     #line2, = ax.plot(x, np.array(test_loss_log), label='test loss')
#     ax.set_xlabel('step')
#     ax.set_ylabel('loss')
#     ax.legend()
#     plt.show()
# except Exception as err:
#     print(err)
