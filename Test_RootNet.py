import tensorflow as tf
from tensorflow.python import keras
from Models import RootNet
from Utils.Config import Config


config = Config()

def Resnet():
    model = keras.applications.resnet50.ResNet50(weights="imagenet",  include_top=False)
    return keras.Model(model.input, model.output)


def MobileNet():
    model = keras.applications.mobilenet.MobileNet(weights="imagenet",  include_top=False)
    return keras.Model(model.input, model.output)


def build_model_RootNet(input, k_value):
    resnet_model = Resnet()
    x = resnet_model(input)
    return RootNet.build_model(input=input, backbone=x, k_value=k_value, config=config)

input = keras.layers.Input(shape=(225, 225, 3))
k_value_input = keras.layers.Input(shape=(None, None, None))


# k_value input shape its just a test at this moment.
model = build_model_RootNet(input, k_value_input)
model.summary()