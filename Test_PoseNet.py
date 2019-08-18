import tensorflow as tf
from tensorflow.python import keras
from Models import PoseNet
from Utils.Config import Config
from Models.MultiPoseRCNN import MultiPoseRCNN


config = Config()

def Resnet():
    model = keras.applications.resnet50.ResNet50(weights="imagenet",  include_top=False)
    return keras.Model(model.input, model.output)


def MobileNet():
    model = keras.applications.mobilenet.MobileNet(weights="imagenet",  include_top=False)
    return keras.Model(model.input, model.output)


def build_model_PoseNet(input):
    resnet_model = Resnet()
    x = resnet_model(input)
    x = light_head(input=x, kernel=4)
    return PoseNet.PoseNet(input, x,  joint_num=config.joint_num, cfg=config)


def light_head(input, kernel=15, padding="valid", bias=False, bn=False):
    k_width = (kernel, 1)
    k_height = (1, kernel)
    x = keras.layers.Conv2D(256, kernel_size=k_width, strides=1,  use_bias=bias, padding=padding)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(490, kernel_size=k_height, strides=1, use_bias=bias, padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, kernel_size=k_width, strides=1, use_bias=bias, padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(490, kernel_size=k_height, strides=1, use_bias=bias, padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(tf.nn.relu)(x)
    return x

input = keras.layers.Input(shape=(225, 225, 3))
model = build_model_PoseNet(input)
model.summary()

