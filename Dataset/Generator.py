import sys
sys.path.append("../")
from Utils.Config import Config
from Dataset.Human36m import Human36M
config = Config()
import numpy as np
import cv2
import random





class Generator(Human36M):
    def __init__(self, data_path, config, mode="Train", shuffle=True):
        super().__init__(data_path=data_path, config=config, mode=mode)
        self.shuffle = shuffle
        self.batch_size = config.batch_size
        self.on_epoch_end()

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]
        data = [self.data[i] for i in indexes]
        return self.get_data_(data)

    def __len__(self):
        return len(self.data)


    def get_data_(self, datas):
        joint_num = self.joint_num
        skeleton = self.skeleton
        flip_pairs = self.flip_pairs
        joints_have_depth = self.joints_have_depth

        for key, data in enumerate(datas):
            bbox = data['bbox']
            joint_img = data['joint_img']
            joint_vis = data['root_vis']
            area = data['area']
            f = data['f']
            joint_img = data['joint_img']


            # 1. load image
            cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if not isinstance(cvimg, np.ndarray):
                raise IOError("Fail to read %s" % data['img_path'])
            img_height, img_width, img_channels = cvimg.shape

            # 2. get augmentation params
            if self.do_augment:
                rot, do_flip, color_scale = self.get_aug_config()
            else:
                rot, do_flip, color_scale = 0, False, [1.0, 1.0, 1.0]



    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # helper functions
    def get_aug_config(self):
        rot_factor = 30
        color_factor = 0.2

        rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= 0.6 else 0
        do_flip = random.random() <= 0.5
        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

        return rot, do_flip, color_scale





Generator(data_path="/home/princemerveil/Downloads", config=config, mode="Train")