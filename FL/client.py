import tensorflow as tf
import numpy as np
import math

from agents.imitation.imitation_learning_network import load_imitation_learning_network
from CIL_train_test import CIL_loss


class Clients:
    def __init__(self, train_data_manager, test_data_manager, learning_rate, clients_num):
        # self.graph = tf.Graph()
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        # -----------------超参数
        self.BATCH_SIZE = 200
        self.image_size = (88, 200, 3)
        self.TARGET_NUM = 3
        self.DROP_OUT_TRAIN = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
        self.DROP_OUT_TEST = [1.0 for i in self.DROP_OUT_TRAIN]
        self.BRANCH_WEI = [0.95, 0.95, 0.95, 0.95, 0.05]
        self.VARIABLE_WEI = {'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05}
        self.LEARNING_RATE = learning_rate
        self.BATCH_NUM_PER_AGENT = 877  # np.floor(总数据量*20%/3)
        self.CLIENTS_NUM = clients_num
        self.GRAPH_LOG = False
        self.global_step = tf.train.get_or_create_global_step()
        self.test_loss_lowest = 100

        # -------------数据集相关
        self.train_data_manager = train_data_manager
        self.test_data_manager = test_data_manager

        self.image_input = tf.placeholder("float", shape=[None, self.image_size[0],
                                                          self.image_size[1], self.image_size[2]], name="input_image")
        self.speed_mea = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, 1], name="speed_mea")
        self.target = tf.placeholder(tf.float32, shape=[None, self.TARGET_NUM], name="target")
        self._input = [None, self.speed_mea]

        self._d_out = tf.placeholder(tf.float32, shape=[len(self.DROP_OUT_TRAIN)])
        self.mask_b1 = tf.placeholder(tf.float32, shape=[None, self.TARGET_NUM], name='follow_lane')
        self.mask_b2 = tf.placeholder(tf.float32, shape=[None, self.TARGET_NUM], name='left')
        self.mask_b3 = tf.placeholder(tf.float32, shape=[None, self.TARGET_NUM], name='right')
        self.mask_b4 = tf.placeholder(tf.float32, shape=[None, self.TARGET_NUM], name='straight')
        self.mask = [self.mask_b1, self.mask_b2, self.mask_b3, self.mask_b4]
        # Call the create function to build the computational graph of AlexNet
        with tf.name_scope("Network"):
            self.branches = load_imitation_learning_network(self.image_input, self._input, self.image_size,
                                                            self._d_out)
        self.loss = CIL_loss.loss_compute(self.mask, self.branches, self.target, self.speed_mea, self.BRANCH_WEI,
                                          self.VARIABLE_WEI, self.BATCH_SIZE)
        self.train_step = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss, global_step=self.global_step)

        # initialize
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

    def run_test(self, test_num, ep):
        test_loss = []
        with self.graph.as_default():
            test_batch_path_list = self.test_data_manager.file_path_list
            for batch_ctr in range(len(test_batch_path_list[0:test_num])):
                test_batch_images_shuffled, test_batch_labels_shuffled, test_file_wrong = \
                    self.test_data_manager.get_shuffled_batch(test_batch_path_list[batch_ctr])

                test_mask_b1, test_mask_b2, test_mask_b3, test_mask_b4 = \
                    self.test_data_manager.command(test_batch_labels_shuffled)

                test_target_in, test_speed_mea_in = self.test_data_manager.get_target_input(test_batch_labels_shuffled)

                feed_dict = {self.image_input: test_batch_images_shuffled,
                             self.speed_mea: test_speed_mea_in,
                             self.target: test_target_in,
                             self.mask_b1: test_mask_b1,
                             self.mask_b2: test_mask_b2,
                             self.mask_b3: test_mask_b3,
                             self.mask_b4: test_mask_b4,
                             self._d_out: self.DROP_OUT_TEST}
                test_loss.append(self.sess.run([self.loss], feed_dict=feed_dict))
                print("epoch"+str(ep+1)+":test_num:", batch_ctr+1)
        return np.mean(test_loss)

    def train_epoch(self, cid, ep):
        """
            Train one client with its own data for one epoch
            cid: Client id
        """
        client_batch_path_list = self.train_data_manager.file_path_list[((cid)*self.BATCH_NUM_PER_AGENT):
                                                                        ((cid+1)*self.BATCH_NUM_PER_AGENT)]
        with self.graph.as_default():
            for batch_ctr in range(len(client_batch_path_list)):
                # ----------------准备数据
                train_batch_images_shuffled, train_batch_labels_shuffled, train_file_wrong = \
                    self.train_data_manager.get_shuffled_batch(client_batch_path_list[batch_ctr])
                if train_file_wrong:
                    continue
                train_mask_b1, train_mask_b2, train_mask_b3, train_mask_b4 = \
                    self.train_data_manager.command(train_batch_labels_shuffled)
                train_target_in, train_speed_mea_in = \
                    self.train_data_manager.get_target_input(train_batch_labels_shuffled)

                feed_dict = {self.image_input: train_batch_images_shuffled,
                             self.speed_mea: train_speed_mea_in,
                             self.target: train_target_in,
                             self.mask_b1: train_mask_b1,
                             self.mask_b2: train_mask_b2,
                             self.mask_b3: train_mask_b3,
                             self.mask_b4: train_mask_b4,
                             self._d_out: self.DROP_OUT_TRAIN}

                if self.GRAPH_LOG:
                    self.GRAPH_LOG = False
                    tf.summary.FileWriter('logs/', self.sess.graph)

                # -----------------------训练
                self.sess.run([self.train_step], feed_dict=feed_dict)
                print("epoch"+str(ep+1)+":client"+str(cid+1)+"训练次数：", batch_ctr+1)

    def get_client_vars(self):
        """ Return all of the variables list """
        with self.graph.as_default():
            client_vars = self.sess.run(tf.trainable_variables())
        return client_vars

    def set_global_vars(self, global_vars):
        """ Assign all of the variables with global vars """
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, global_vars):
                variable.load(value, self.sess)

    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients """
        client_num = self.CLIENTS_NUM
        choose_num = math.ceil(client_num * ratio)
        return np.random.permutation(client_num)[:choose_num]

