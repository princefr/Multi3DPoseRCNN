import tensorflow as tf
from tensorflow.python import keras




def DeconvLayers(input):
    # HEAD DÉCONVOLUTION
    x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu, kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(input)
    x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu, kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(x)
    x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu, kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(x)
    return x


def TensorSumTrue(data):
    a, axis = data
    return keras.backend.sum(a, axis=axis, keepdims=True) -1

def TensorSumFalse(data):
    a, axis = data
    return keras.backend.sum(a, axis=axis)

def multiply_one(data):
    a, config = data
    b = keras.backend.cast(tf.keras.backend.arange(1, config.output_shape_pose[1] + 1), tf.float32)[0]
    return a * b

def multiply_two(data):
    a, config = data
    b = keras.backend.cast(keras.backend.arange(1, config.output_shape_pose[0] + 1), tf.float32)[0]
    return a * b

def multiply_tree(data):
    a, config = data
    b = keras.backend.cast(keras.backend.arange(1, config.depth_dim + 1), tf.float32)[0]
    return a * b

def FinalLayer(input, joint_num , cfg):
    x = keras.layers.Conv2D(joint_num * cfg.depth_dim, kernel_size=1, strides=1, padding='valid')(input)
    return x


def softmax(data):
    x, axis = data
    return keras.backend.softmax(x, axis=axis)

def PoseNet(input, backbone, joint_num, cfg):
    x = DeconvLayers(backbone)
    heatmaps = FinalLayer(x, joint_num, cfg)

    heatmaps = keras.layers.Reshape((joint_num, cfg.depth_dim * cfg.output_shape_pose[0] * cfg.output_shape_pose[1]))(heatmaps)
    heatmaps = keras.layers.Softmax(axis=2)(heatmaps)
    heatmaps = keras.layers.Reshape((joint_num, cfg.depth_dim, cfg.output_shape_pose[0], cfg.output_shape_pose[1]))(heatmaps)


    accu_x = keras.layers.Lambda(TensorSumFalse)((heatmaps, (2, 3)))
    accu_y = keras.layers.Lambda(TensorSumFalse)((heatmaps, (2, 4)))
    accu_z = keras.layers.Lambda(TensorSumFalse)((heatmaps, (3, 4)))

    accu_x = keras.layers.Lambda(multiply_one)((accu_x, cfg))
    accu_y = keras.layers.Lambda(multiply_two)((accu_y, cfg))
    accu_z = keras.layers.Lambda(multiply_tree)((accu_z, cfg))

    accu_x = keras.layers.Lambda(TensorSumTrue)((accu_x, 2))
    accu_y = keras.layers.Lambda(TensorSumTrue)((accu_y, 2))
    accu_z = keras.layers.Lambda(TensorSumTrue)((accu_z, 2))

    coord_out = keras.layers.Concatenate(axis=2)([accu_x, accu_y, accu_z])



    return keras.Model(input, coord_out)

