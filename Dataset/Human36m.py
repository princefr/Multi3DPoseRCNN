import copy
import numpy as np
from tensorflow.python import keras
import tensorflow as tf
import time
import random
import sys
import os.path as osp
from pycocotools.coco import COCO
sys.path.append("../")
import math
from Utils.Config import Config
import cv2
import json
import ijson






class Human36M(keras.utils.Sequence):
    def __init__(self, data_path, config, mode="Train"):
        assert "Train" or "Validation" in mode
        self.img_dir = osp.join(data_path, 'data', 'images')
        self.annot_dir = osp.join(data_path, 'data',  'annotations')
        self.joint_num = 18
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
        self.root_idx = self.joints_name.index('Pelvis')
        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')
        self.min_depth = 2000
        self.max_depth = 8000
        self.joints_have_depth = True
        self.protocol = 1
        self.config = config

        self.flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        self.skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        self.joints_have_depth = True
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 16) # except Thorax
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                            'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
                            'WalkTogether']
        if mode == "Train":
            self.subject_list = [1, 5, 6, 7, 8]
            self.sampling_ratio = 5
        else:
            self.subject_list = [9, 11]
            self.sampling_ratio = 64

        self.data = self.load_data()

    def load_data(self):
        db = COCO()
        for subject in self.subject_list:
            with open(osp.join(self.annot_dir, 'Human36M_subject' + str(subject) + '.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
        db.createIndex()



        if self.mode == 'Validation' and not self.config.use_gt_bbox:
            print("Get bounding box from " + self.human_bbox_dir)
            bbox_result = {}
            with open(self.human_bbox_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_result[str(annot[i]['image_id'])] = np.array(annot[i]['bbox'])
        else:
            print("Get bounding box from groundtruth")

        # get the data
        data = []
        for aid in db.anns.keys():
            ann = db.anns[aid]

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]

            # check subject and frame_idx
            subject = img['subject'];
            frame_idx = img['frame_idx'];
            if subject not in self.subject_list:
                continue
            if frame_idx % self.sampling_ratio != 0:
                continue

            img_path = osp.join(self.img_dir, img['file_name'])
            img_width, img_height = img['width'], img['height']
            cam_param = img['cam_param']
            R, t, f, c = np.array(cam_param['R']), np.array(cam_param['t']), np.array(cam_param['f']), np.array(
                cam_param['c'])

            # project world coordinate to cam, image coordinate
            root_world = np.array(ann['keypoints_world'])[self.root_idx]
            root_cam = np.array(ann['keypoints_cam'])[self.root_idx]
            root_img = np.array(ann['keypoints_img'])[self.root_idx]

            # image - rootdepth
            joint_img = np.concatenate((root_img, root_cam[:,:2]),1)
            joint_img[:,2] = joint_img[:,2] - root_cam[self.root_idx,2]

            # image with normal depth
            root_img = np.concatenate([root_img, root_cam[2:3]])
            root_vis = np.array(ann['keypoints_vis'])[self.root_idx, None]

            if self.data_split == 'test' and not self.config.use_gt_bbox:
                bbox = bbox_result[str(image_id)]
            else:
                bbox = np.array(ann['bbox'])

            # sanitize bboxes
            x, y, w, h = bbox
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
            if w * h > 0 and x2 >= x1 and y2 >= y1:
                bbox = np.array([x1, y1, x2 - x1, y2 - y1])
            else:
                continue

            # aspect ratio preserving bbox
            w = bbox[2]
            h = bbox[3]
            c_x = bbox[0] + w / 2.
            c_y = bbox[1] + h / 2.
            aspect_ratio = self.config.input_shape[1] / self.config.input_shape[0]
            if w > aspect_ratio * h:
                h = w / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio
            bbox[2] = w * 1.25
            bbox[3] = h * 1.25
            bbox[0] = c_x - bbox[2] / 2.
            bbox[1] = c_y - bbox[3] / 2.
            area = bbox[2] * bbox[3]

            data.append({
                'img_path': img_path,
                'img_id': image_id,
                'bbox': bbox,
                'area': area,
                'root_img': root_img,  # [org_img_x, org_img_y, depth]
                'root_cam': root_cam,  # [X, Y, Z] in camera coordinate
                'root_vis': root_vis,
                'joint_img' : joint_img,  # [org_img_x, org_img_y, depth - root_depth]
                'f': f,
                'c': c
            })
        return data


