import h5py
import os
import numpy as np

import matplotlib.pyplot as plt
import imgaug.augmenters as iaa


class DataManager:
    def __init__(self, data_path, target_num=3, training_mode=False):
        self.data_path = data_path
        self.file_path_list = self.get_data()
        self.target_num = target_num
        self.training_mode = training_mode

    def get_data(self):
        """
        获取数据集中的batches路径
        :return: batches路径
        """
        file_list = os.listdir(self.data_path)
        file_path_list = []
        for file in file_list:
            file_path_list.append(self.data_path+'/'+file)
        return file_path_list

    def shuffle_data(self, batch_num=None):
        """
        随机选取batch_num个mini_batch
        :param batch_num: 一个智能体使用的batch个数
        :return: 打乱的batches路径列表
        """
        np.random.shuffle(self.file_path_list)
        if batch_num is not None:
            return self.file_path_list[:batch_num]
        else:
            return self.file_path_list

    @staticmethod
    def get_shuffled_batch(h5file_path, file_wrong_button=False):
        """
        将batch内部打乱
        :param h5file_path: batch路径
        :param file_wrong_button: 防止文件不存在导致的错误
        :return: 内部打乱的batch数据、标签和文件错误标志
        """
        try:
            batch = h5py.File(h5file_path, mode='r')
            images = list(batch['rgb'])
            labels = list(batch['targets'])
            tem = np.array([images, labels], dtype=object)
            tem = tem.transpose()
            np.random.shuffle(tem)
            # TODO: 将图片归一化或伸缩至[0,1]范围内
            images_shuffled = np.array([np.array(item) for item in tem[:, 0]])  # * 1. / 255.
            labels_shuffled = np.array([np.array(item) for item in tem[:, 1]])
            return images_shuffled, labels_shuffled, file_wrong_button
        except OSError:
            print('wrong_file:', h5file_path)
            file_wrong_button = True
            return None, None, file_wrong_button

    def command(self, labels_shuffled):
        """
        :param labels_shuffled: 内部打乱的batch标签, type: np.array
        :return: command: type: np.array, shape: (batch_size * 1)
        """
        # when command = 2, branch 1 (follow lane) is activated
        command = labels_shuffled[:, 24:25]  # 为了保持维度
        mask_b1 = (command == 2)
        mask_b1 = np.hstack([mask_b1] * self.target_num)
        mask_b2 = (command == 3)
        mask_b2 = np.hstack([mask_b2] * self.target_num)
        mask_b3 = (command == 4)
        mask_b3 = np.hstack([mask_b3] * self.target_num)
        mask_b4 = (command == 5)
        mask_b4 = np.hstack([mask_b4] * self.target_num)
        return mask_b1, mask_b2, mask_b3, mask_b4

    @staticmethod
    def get_target_input(labels_shuffled):
        """
        :param labels_shuffled: 内部打乱的batch标签, type: np.array
        :return: target: type: np.array, shape: (batch_size * target_num)
                 speed_mea: type: np.array, shape: (batch_size * 1)
        """
        target = labels_shuffled[:, :3]
        # TODO: 查看标签数据中速度有没有被归一化->没有被归一化
        speed_mea = labels_shuffled[:, 10:11] / 25

        return target, speed_mea

    def get_aug_img(self, batch_image):
        """
        :param batch_image: batch_size张图片；type:np.array
        :return: aug_img or original images
        """
        if self.training_mode:
            aug_img = image_augment(batch_image)
            return aug_img * 1. / 255.
        else:
            return batch_image * 1. / 255.


def image_augment(batch):
    seq = iaa.Sequential([
        iaa.Sometimes(0.09, iaa.GaussianBlur(sigma=(0.0, 1.5))),
        iaa.Sometimes(0.09, iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0.0, 0.5**2), per_channel=True),
                                          iaa.AdditiveGaussianNoise(scale=(0.0, 0.5**2), per_channel=False))),
        iaa.Sometimes(0.30, iaa.Dropout((0.0, 0.1), per_channel=0.5)),
        iaa.Sometimes(0.30, iaa.CoarseDropout(size_percent=(0.08, 0.2), per_channel=0.5)),
        iaa.Sometimes(0.30, iaa.LinearContrast((0.5, 2.0), per_channel=0.5))
    ])
    aug_img = seq(images=batch)
    return aug_img


if __name__ == "__main__":
    Data_Path = r'D:\soft\Desktop\image_aug\SeqTra'
    data_manager = DataManager(Data_Path, training_mode=True)
    images, labels, _ = data_manager.get_shuffled_batch(data_manager.file_path_list[0])
    for image in images[:3]:
        plt.imshow(image)
        plt.show()

    aug_img = data_manager.get_aug_img(images)
    for image in aug_img[:3]:
        plt.imshow(image)
        plt.show()
