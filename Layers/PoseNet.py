import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
import numpy as np
import sys
sys.path.append("../")
from Layers.RoiPooling import PyramidROIAlign



def TensorSumTrue(data):
    a, axis = data
    return keras.backend.sum(a, axis=axis, keepdims=True) -1

def TensorSumFalse(data):
    a, axis = data
    return keras.backend.sum(a, axis=axis)

def multiply_one(data):
    a, config = data
    b = keras.backend.cast(tf.keras.backend.arange(1, config.output_shape[1] + 1), tf.float32)[0]
    return a * b

def multiply_two(data):
    a, config = data
    b = keras.backend.cast(keras.backend.arange(1, config.output_shape[0] + 1), tf.float32)[0]
    return a * b

def multiply_tree(data):
    a, config = data
    b = keras.backend.cast(keras.backend.arange(1, config.depth_dim + 1), tf.float32)[0]
    return a * b



class PoseNet():
    def __init__(self,
                 rois,
                 feature_maps,
                 pool_size,
                 image_shape,
                 num_classes,
                 max_regions,
                 fc_layers_size,
                 pyramid_top_down_size, joint_num, depth_dim, config):
        print("hui")
        self.rois = rois
        self.feature_maps = feature_maps
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.max_regions = max_regions
        self.fc_layers_size = fc_layers_size
        self.pyramid_top_down_size = pyramid_top_down_size
        self.joint_num = joint_num
        self.depth_dim = depth_dim
        self.config = config


    def build(self):
        rois = self.rois
        feature_maps = self.feature_maps
        pool_size = self.pool_size
        image_shape = self.image_shape
        pyramid = PyramidROIAlign(name="roi_align_PoseNet", pool_shape=[pool_size, pool_size], image_shape=image_shape)([rois] + feature_maps)
        x = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu,
                                         kernel_initializer=keras.initializers.truncated_normal(0, 0.01)))(pyramid)
        x = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu,
                                         kernel_initializer=keras.initializers.truncated_normal(0, 0.01)))(x)
        x = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu,
                                         kernel_initializer=keras.initializers.truncated_normal(0, 0.01)))(x)
        heatmaps = keras.layers.Conv2D(self.joint_num * self.depth_dim, kernel_size=1, strides=1, padding='valid')(x)

        heatmaps = keras.layers.Reshape((self.joint_num, self.config.depth_dim * self.config.output_shape_pose[0] * self.config.output_shape_pose[1]))(heatmaps)
        heatmaps = keras.layers.Softmax(axis=2)(heatmaps)
        heatmaps = keras.layers.TimeDistributed(keras.layers.Reshape((self.joint_num, self.config.depth_dim, self.config.output_shape_pose[0], self.config.output_shape_pose[1])))(heatmaps)

        accu_x = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumFalse))((heatmaps, (2, 3)))
        accu_y = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumFalse))((heatmaps, (2, 4)))
        accu_z = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumFalse))((heatmaps, (3, 4)))

        accu_x = keras.layers.TimeDistributed(keras.layers.Lambda(multiply_one))((accu_x, self.config))
        accu_y = keras.layers.TimeDistributed(keras.layers.Lambda(multiply_two))((accu_y, self.config))
        accu_z = keras.layers.TimeDistributed(keras.layers.Lambda(multiply_tree))((accu_z, self.config))

        accu_x = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumTrue))((accu_x, 2))
        accu_y = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumTrue))((accu_y, 2))
        accu_z = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumTrue))((accu_z, 2))

        coord_out = keras.layers.TimeDistributed(keras.layers.Concatenate(axis=2))([accu_x, accu_y, accu_z])

        return coord_out, None

    def build_training(self):
        feature_maps = self.feature_maps
        x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu, kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(feature_maps)
        x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu, kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(x)
        x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu, kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(x)

        heatmaps = keras.layers.Conv2D(self.joint_num * self.config.depth_dim, kernel_size=1, strides=1, padding='valid')(x)

        heatmaps = keras.layers.Reshape((self.joint_num, self.config.depth_dim * self.config.output_shape_pose[0] * self.config.output_shape_pose[1]))(heatmaps)
        heatmaps = keras.layers.Softmax(axis=2)(heatmaps)
        heatmaps = keras.layers.Reshape((self.config.joint_num, self.config.depth_dim, self.config.output_shape_pose[0], self.config.output_shape_pose[1]))(heatmaps)

        accu_x = keras.layers.Lambda(TensorSumFalse)((heatmaps, (2, 3)))
        accu_y = keras.layers.Lambda(TensorSumFalse)((heatmaps, (2, 4)))
        accu_z = keras.layers.Lambda(TensorSumFalse)((heatmaps, (3, 4)))

        accu_x = keras.layers.Lambda(multiply_one)((accu_x, self.config))
        accu_y = keras.layers.Lambda(multiply_two)((accu_y, self.config))
        accu_z = keras.layers.Lambda(multiply_tree)((accu_z, self.config))

        accu_x = keras.layers.Lambda(TensorSumTrue)((accu_x, 2))
        accu_y = keras.layers.Lambda(TensorSumTrue)((accu_y, 2))
        accu_z = keras.layers.Lambda(TensorSumTrue)((accu_z, 2))

        coord_out = keras.layers.Concatenate(axis=2)([accu_x, accu_y, accu_z])

        return keras.Model(input, coord_out)