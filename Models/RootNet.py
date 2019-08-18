import tensorflow as tf
from tensorflow.python import keras


def Resnet():
    model = keras.applications.resnet50.ResNet50(weights="imagenet",  include_top=False)
    return keras.Model(model.input, model.output)

def MobileNet():
    model = keras.applications.mobilenet.MobileNet(weights="imagenet",  include_top=False)
    return keras.Model(model.input, model.output)

def Xception():
    model = keras.applications.xception.Xception(weights="imagenet",  include_top=False)
    return keras.Model(model.input, model.output)

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

def squeeze(input):
    return keras.backend.squeeze(input, axis=1)

def build_model(input, backbone, config, k_value):
    # HEAD DÉCONVOLUTION LAYERS
    x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu, kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(backbone)
    x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu, kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(x)
    x = keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=2, padding="same", activation=tf.nn.relu, kernel_initializer=keras.initializers.truncated_normal(0, 0.01))(x)

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
    z = keras.layers.Reshape((backbone.shape[-1], backbone.shape[1] * backbone.shape[2]))(backbone)
    z = keras.layers.Lambda(prepare_depth_input, name="prepare_dataset")(z)
    z = keras.layers.Lambda(unsqueeze, name="unsqueeze_for_the_fisrt")((z, 2))
    z = keras.layers.Lambda(unsqueeze, name="unsqueeze_for_the_second")((z, 3))
    z = keras.layers.Reshape((z.shape[-2], z.shape[-1], z.shape[1]))(z)

    # used to calculate the z cordinatete
    depth_map = keras.layers.Conv2D(1, kernel_size=1, strides=1, padding="valid", name="z_cordinate")(z)
    z = keras.layers.Reshape((-1, depth_map.shape[2]))(depth_map)
    z = keras.layers.Lambda(squeeze)(z)
    z = keras.layers.Lambda(multiply)([z, k_value])

    return keras.Model([input, k_value], [coord_x, coord_y,  z])

