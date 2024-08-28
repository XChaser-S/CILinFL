import os
import tensorflow as tf
from tqdm import tqdm

from client import Clients
from CIL_train_test.data import DataManager

import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" #调用编号为0的GPU


def build_clients(num, train_data_manager, test_data_manager):
    learning_rate = 0.0001
    # create Client and model
    return Clients(train_data_manager,
                   test_data_manager,
                   learning_rate=learning_rate,
                   clients_num=num)


def run_global_test(client, global_vars, test_num, ep):
    client.set_global_vars(global_vars)
    loss = client.run_test(test_num, ep)
    print("[epoch {}] Testing , Loss: {:.4f}".format(ep + 1,  loss))
    return loss
    # print("[epoch {}, {} batches] Testing , Loss: {:.4f}".format(ep + 1, test_num, loss))

#### SOME TRAINING PARAMS ####
LOG_STEP_GAP = 12
LOSS_MIN = 100
PRE_TRAINED_MODEL_PATH = tf.train.latest_checkpoint(r'log7')
CLIENT_NUMBER = 1
CLIENT_RATIO_PER_ROUND = 0.12
epoch = 96

train_data_path = r'/home/mist/AgentHuman/SeqTrain'
test_data_path = r'/home/mist/AgentHuman/SeqVal'
# train_data_path = r'D:\database\CORL2017ImitationLearningData\CORL2017ImitationLearningData\AgentHuman\local'
# test_data_path = r'D:\database\CORL2017ImitationLearningData\CORL2017ImitationLearningData\AgentHuman\SeqVal'

train_data_manager = DataManager(train_data_path, 3)
test_data_manager = DataManager(test_data_path, 3)
loss_total = []

# ### CREATE CLIENT AND LOAD DATASET ####
client = build_clients(CLIENT_NUMBER, train_data_manager, test_data_manager)

client_saver = tf.train.Saver()
best_saver = tf.train.Saver(max_to_keep=3)
# client_saver = tf.train.Saver(max_to_keep=30)
# ### BEGIN TRAINING ####
# ToDo:Restore parameter and get it with a list or array

client_saver.restore(client.sess, PRE_TRAINED_MODEL_PATH)
print(PRE_TRAINED_MODEL_PATH)

global_vars = client.get_client_vars()
for ep in range(epoch):
    # We are going to sum up active clients' vars at each epoch
    client_vars_sum = None

    # Choose some clients that will train on this epoch
    # random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)

    # Train with these clients
    for client_id in tqdm(range(CLIENT_NUMBER)):
        # Restore global vars to client's model
        client.set_global_vars(global_vars)

        # train one client
        client.train_epoch(client_id, ep)  # 讨论1
        # if (ep+1) % LOG_STEP_GAP == 0:
        #     client_saver.save(client.sess, 'log10/client/client{}_model'.format(client_id),
        #                       global_step=ep+1, write_meta_graph=False)
        # obtain current client's vars
        current_client_vars = client.get_client_vars()

        # sum it up
        if client_vars_sum is None:
            client_vars_sum = current_client_vars
        else:
            for cv, ccv in zip(client_vars_sum, current_client_vars):
                cv += ccv

    # obtain the avg vars as global vars
    global_vars = []
    for var in client_vars_sum:
        global_vars.append(var / CLIENT_NUMBER)

    # ToDo: get averaged loss on test dataset
    each_ep_loss = run_global_test(client, global_vars, 374, ep)
    # each_ep_loss = run_global_test(client, global_vars)

    print("epoch"+str(ep+1)+"_global_loss:", each_ep_loss)
    loss_total.append(each_ep_loss)
    if each_ep_loss < LOSS_MIN:
        LOSS_MIN = each_ep_loss
        best_saver.save(client.sess, 'log10/FL_best/client_model', global_step=ep+1, write_meta_graph=False)
    if (ep+1) % LOG_STEP_GAP == 0:
        client_saver.save(client.sess, 'log10/FL/client_model', global_step=ep + 1, write_meta_graph=False)
np.save('log10/loss/client_loss.npy', loss_total)
    # print('running1')
    # # run test on 600 instances
    # run_global_test(client, global_vars, test_num=5, ctr=1)
    # print('running2')


# # print("running3")
# #### FINAL TEST ####
# final_global_loss=run_global_test(client, global_vars, test_num=374)
# print("final_global_loss:",final_global_loss)