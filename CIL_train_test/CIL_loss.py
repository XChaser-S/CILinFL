import tensorflow as tf
import numpy as np


def loss_branch_lst(branches, masks, target, speed_mea, branch_wei):
    """
    :param branches: type: list, item_type: tf.tensor, shape: (branch_num * batch_size * (target_num + speed))
    :param masks: type: list, item_type: tf.tensor, shape: (branch_num * batch_size * target_num)
    :param target: type: tf.tensor, shape: (batch_size * target_num)
    :param speed_mea: type: tf.tensor, shape: (batch_size * 1)
    :param branch_wei: type: list, item_type: float, shape: (branch_num + speed_branch)
    :return: loss_branch_list: type: list, item_type: tf.tensor, shape: (branch_num * batch_size * (target_num + speed))
    """
    with tf.name_scope('loss_branches'):
        loss_branch_list = []
        # TODO This is hardcoded but all our cases rigth now uses four branches
        for i in range(len(branches) - 1):
            loss_branch_list.append(((branches[i] - target) ** 2 * masks[i]) * branch_wei[i])
        """ The last branch is a speed branch"""
        # TODO: Activate or deactivate speed branch loss
        loss_branch_list.append((branches[-1] - speed_mea) ** 2 * branch_wei[-1])
    return loss_branch_list


def loss_compute(mask, branches, target, speed_mea, branch_wei, variable_wei, batch_size):
    """
    :param mask: type: list, shape: (4), item_type: tf.placeholder item_shape: (batch_size, target_num)
    :param branches: type: list, item_type: tf.tensor, shape: (branch_num * batch_size * (target_num + speed))
    :param target: type: tf.tensor, shape: (batch_size * target_num)
    :param speed_mea: type: tf.tensor, shape: (batch_size * 1)
    :param branch_wei: type: list, item_type: float, shape: (branch_num + speed_branch)
    :param variable_wei: type: dict, item_num: target_num
    :param batch_size: int
    :return: loss: type: tf.tensor, shape: (1)
    """
    with tf.name_scope('loss'):
        loss_branch_list = loss_branch_lst(branches, mask, target, speed_mea, branch_wei)
        print(np.shape(loss_branch_list[0]))
        for i in range(4):
            loss_branch_list[i] = loss_branch_list[i][:, 0] * variable_wei['Steer'] \
                                   + loss_branch_list[i][:, 1] * variable_wei['Gas'] \
                                   + loss_branch_list[i][:, 2] * variable_wei['Brake']

        loss_function = loss_branch_list[0] + loss_branch_list[1] + loss_branch_list[2] + loss_branch_list[3]

        speed_loss = loss_branch_list[4]/batch_size
        loss_result = tf.add(tf.reduce_sum(loss_function) / batch_size,
                             tf.reduce_sum(speed_loss) / batch_size, name='loss_result')
    return loss_result


if __name__ == '__main__':
    import random
    branch_wei = [0.95, 0.95, 0.95, 0.95, 0.05]
    variable_wei = {'Steer': 0.5, 'Gas': 0.45, 'Brake': 0.05}
    command = tf.constant(np.random.randint(2,5,[10, 1]), name="command")
    steer = np.random.random([10]) * 2 - 1
    gas = np.random.random([10])
    brake = np.random.random([10])

    branch = np.array([steer, gas, brake], dtype=object)
    branch = branch.transpose()
    speed = tf.convert_to_tensor(np.random.random([10, 1]) * 50, dtype=tf.float32, name='speed')
    print(np.shape(branch))

    branch0 = tf.convert_to_tensor(branch, dtype=tf.float32,name='branch0')
    branch1 = tf.convert_to_tensor(branch, dtype=tf.float32, name='branch1')
    branch2 = tf.convert_to_tensor(branch, dtype=tf.float32, name='branch2')
    branch3 = tf.convert_to_tensor(branch, dtype=tf.float32, name='branch3')
    branch = [branch0, branch1, branch2, branch3, speed]

    steer_target = np.random.random([10]) * 2 - 1
    gas_target = np.random.random([10])
    brake_target = np.random.random([10])
    target = np.array([steer_target, gas_target, brake_target], dtype=object)
    target = target.transpose()
    target = tf.convert_to_tensor(target, dtype=tf.float32, name='target')
    speed_mea = tf.convert_to_tensor(np.random.random([10, 1]) * 50, dtype=tf.float32, name='speed_mea')
    loss = loss_compute(command,  branch, 3, target, speed_mea, branch_wei, variable_wei)
    tf.summary.FileWriter(r'E:\毕业设计\5论文代码\CIL\log\c_value1', tf.get_default_graph())
    with tf.Session() as sess:
        print(sess.run(loss))
        print(sess.run(loss))