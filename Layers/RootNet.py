import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
import numpy as np
import sys
sys.path.append("../")
from Layers.RoiPooling import PyramidROIAlign



def prepare_depth_input(input):
    return keras.backend.mean(input, axis=2)


def unsqueeze(data):
    input, axis = data
    return tf.expand_dims(input, axis=axis)


def multiply(input):
    input_a, input_b = input
    return input_a * keras.backend.reshape(input_b, (input_a.shape[-1], input_a.shape[-1]))


def TensorSumFalse(data):
    a, axis = data
    return keras.backend.sum(a, axis=(axis))

def TensorSumFalseMinus(data):
    a, axis = data
    return keras.backend.sum(a, axis=axis) -1

def multiply_one(data):
    a, config = data
    b = keras.backend.cast(tf.keras.backend.arange(1, config.output_shape_pose[1] + 1), tf.float32)[0]
    return a * b

def multiply_two(data):
    a, config = data
    b = keras.backend.cast(keras.backend.arange(1, config.output_shape_pose[0] + 1), tf.float32)[0]
    return a * b

def softmax(data):
    x, axis = data
    return keras.backend.softmax(x, axis=axis)


class RootNet():
    def __init__(self,
                 rois,
                 feature_maps,
                 pool_size,
                 image_shape,
                 num_classes,
                 max_regions,
                 fc_layers_size,
                 pyramid_top_down_size, joint_num, depth_dim, config, k_value):
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
        self.k_value = k_value

    def build(self):
        print("this")
        rois = self.rois
        feature_maps = self.feature_maps
        pool_size = self.pool_size
        num_classes = self.num_classes
        image_shape = self.image_shape
        fc_layers_size = self.fc_layers_size

        pyramid = PyramidROIAlign(name="roi_align_PoseNet", pool_shape=[pool_size, pool_size], image_shape=image_shape)([rois] + feature_maps)
        x = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu,
                                         kernel_initializer=keras.initializers.truncated_normal(0, 0.01)))(pyramid)
        x = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu,
                                         kernel_initializer=keras.initializers.truncated_normal(0, 0.01)))(x)
        x = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu,
                                         kernel_initializer=keras.initializers.truncated_normal(0, 0.01)))(x)

        # used to calculate the x and y cordinate
        xy_layer = keras.layers.TimeDistributed(keras.layers.Conv2D(1, kernel_size=1, strides=1, padding="valid"))(x)
        xy_layer = keras.layers.TimeDistributed(keras.layers.Reshape((1, xy_layer.shape[1], self.config.output_shape_pose[0] * self.config.output_shape_pose[1])))(xy_layer)
        xy_layer = keras.layers.TimeDistributed(keras.layers.Softmax(axis=2))(xy_layer)
        xy_layer = keras.layers.TimeDistributed(keras.layers.Reshape((1, self.config.output_shape_pose[0], self.config.output_shape_pose[1])))(xy_layer)

        hm_x = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumFalse))((xy_layer, 2))
        hm_y = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumFalse))((xy_layer, 3))

        coord_x = keras.layers.TimeDistributed(keras.layers.Lambda(multiply_one))((hm_x, self.config))
        coord_y = keras.layers.TimeDistributed(keras.layers.Lambda(multiply_two))((hm_y, self.config))

        coord_x = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumFalseMinus))((coord_x, 2))
        coord_y = keras.layers.TimeDistributed(keras.layers.Lambda(TensorSumFalseMinus))((coord_y, 2))

        # calculate the z coordinate
        z = keras.layers.TimeDistributed(keras.layers.Reshape((pyramid.shape[-1], pyramid.shape[1] * pyramid.shape[2])))(pyramid)
        z = keras.layers.TimeDistributed(keras.layers.Lambda(prepare_depth_input, name="prepare_dataset"))(z)
        z = keras.layers.TimeDistributed(keras.layers.Lambda(unsqueeze, name="unsqueeze_for_the_fisrt"))((z, 2))
        z = keras.layers.TimeDistributed(keras.layers.Lambda(unsqueeze, name="unsqueeze_for_the_second"))((z, 3))
        z = keras.layers.TimeDistributed(keras.layers.Reshape((z.shape[-2], z.shape[-1], z.shape[1])))(z)

        # used to calculate the z cordinatete
        depth_map = keras.layers.TimeDistributed(keras.layers.Conv2D(1, kernel_size=1, strides=1, padding="valid", name="z_cordinate"))(z)
        z = keras.layers.TimeDistributed(keras.layers.Reshape((-1, depth_map.shape[0])))(depth_map)
        z = keras.layers.TimeDistributed(keras.layers.Lambda(multiply))([z, self.k_value])

        return (coord_x, coord_y, z), None


    def build_training(self):
        print("building model for training")
        feature_maps = self.feature_maps
        k_value = self.k_value
        config = self.config
        x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu,
                                         kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(feature_maps)
        x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu,
                                         kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(x)
        x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu,
                                         kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(x)


        # used to calculate the x and y cordinate
        xy_layer = keras.layers.Conv2D(1, kernel_size=1, strides=1, padding="valid")(x)
        xy_layer = keras.layers.Reshape((1, config.output_shape_pose[0] * config.output_shape_pose[1]))(xy_layer)
        xy_layer = keras.layers.Softmax(axis=2)(xy_layer)
        xy_layer = keras.layers.Reshape((1, config.output_shape_pose[0], config.output_shape_pose[1]))(xy_layer)

        hm_x = keras.layers.Lambda(TensorSumFalse)((xy_layer, 2))
        hm_y = keras.layers.Lambda(TensorSumFalse)((xy_layer, 3))

        coord_x = keras.layers.Lambda(multiply_one)((hm_x, config))
        coord_y = keras.layers.Lambda(multiply_two)((hm_y, config))

        coord_x = keras.layers.Lambda(TensorSumFalseMinus)((coord_x, 2))
        coord_y = keras.layers.Lambda(TensorSumFalseMinus)((coord_y, 2))

        # calculate the z coordinate
        z = keras.layers.Reshape((feature_maps.shape[-1], feature_maps.shape[1] * feature_maps.shape[2]))(
            feature_maps)
        z = keras.layers.Lambda(prepare_depth_input, name="prepare_dataset")(z)
        z = keras.layers.Lambda(unsqueeze, name="unsqueeze_for_the_fisrt")((z, 2))
        z = keras.layers.Lambda(unsqueeze, name="unsqueeze_for_the_second")((z, 3))
        z = keras.layers.Reshape((z.shape[-2], z.shape[-1], z.shape[1]))(z)

        # used to calculate the z cordinatete
        depth_map = keras.layers.Conv2D(1, kernel_size=1, strides=1, padding="valid", name="z_cordinate")(z)
        z = keras.layers.Reshape((-1, depth_map.shape[2]))(depth_map)
        z = keras.layers.Lambda(multiply)([z, k_value])

        return keras.Model([input, k_value], [coord_x, coord_y, z])