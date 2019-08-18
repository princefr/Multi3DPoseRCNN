import os
import os.path as osp
import sys
import numpy as np


class Config():
    def __init__(self):
        self.architecture = 'resnet50'
        self.input_width = 256
        self.input_height = 256
        self.input_shape = (self.input_width, self.input_height, 3)
        self.trainset = ['Human36m']
        self.testset = 'Human36m'
        self.resnet_type = 50



        # Pose assiociated config
        self.output_shape_pose = (self.input_shape[0] // 4, self.input_shape[1] // 4)
        self.depth_dim = 64
        self.bbox_3d_shape = (2000, 2000, 2000)
        self.pixel_mean = (0.485, 0.456, 0.406)
        self.pixel_std = (0.229, 0.224, 0.225)
        self.joint_num = 17


        # detection associated config
        self.num_classes = 1 + 80
        self.pre_nms_max_proposals = 6000
        self.max_proposals = 1000
        self.max_detections = 100
        self.pyramid_top_down_size = 256
        self.proposal_nms_threshold = 0.7
        self.detection_min_confidence = 0.7
        self.detection_nms_threshold = 0.3
        self.bounding_box_std_dev = [0.1, 0.1, 0.2, 0.2]
        self.classifier_pool_size = 7
        self.mask_pool_size = 14
        self.fc_layers_size = 1024
        self.anchor_scales = (32, 64, 128, 256, 512)
        self.anchor_ratios = [0.5, 1, 2]
        self.anchors_per_location = len(self.anchor_ratios)
        self.backbone_strides = [4, 8, 16, 32, 64]
        self.anchor_stride = 1




        #training_ config
        self.lr_dec_epoch = [17, 21]
        self.end_epoch = 25
        self.lr = 1e-3
        self.lr_dec_factor = 10
        self.batch_size = 32



        # testing config
        self.test_batch_size = 16
        self.flip_test = True
        self.use_gt_info = False