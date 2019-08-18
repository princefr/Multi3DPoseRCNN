import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
import numpy as np
import sys
sys.path.append("../")
from Layers.RoiPooling import PyramidROIAlign



class FPNClassifier():

    def __init__(self,
                 rois,
                 feature_maps,
                 pool_size,
                 image_shape,
                 num_classes,
                 max_regions,
                 fc_layers_size,
                 pyramid_top_down_size):

        self.rois = rois
        self.feature_maps = feature_maps
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.max_regions = max_regions
        self.fc_layers_size = fc_layers_size
        self.pyramid_top_down_size = pyramid_top_down_size

    def build(self):

        rois = self.rois
        feature_maps = self.feature_maps

        pool_size = self.pool_size
        num_classes = self.num_classes
        image_shape = self.image_shape
        fc_layers_size = self.fc_layers_size

        pyramid = PyramidROIAlign(name="roi_align_classifier",
                                  pool_shape=[pool_size, pool_size],
                                  image_shape=image_shape)([rois] + feature_maps)


        x = keras.layers.TimeDistributed(
            keras.layers.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
            name="mrcnn_class_conv1")(pyramid)
        x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(), name='mrcnn_class_bn1')(x,
                                                                                                    training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.TimeDistributed(keras.layers.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
        x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(), name='mrcnn_class_bn2')(x,
                                                                                                    training=False)
        shared = keras.layers.Activation('relu')(x)
        shared = keras.layers.Lambda(lambda x: keras.backend.squeeze(keras.backend.squeeze(x, 3), 2),
                                     name="pool_squeeze")(shared)
        logits = keras.layers.TimeDistributed(keras.layers.Dense(num_classes), name='mrcnn_class_logits')(shared)
        probabilities = keras.layers.TimeDistributed(keras.layers.Activation("softmax"), name="mrcnn_class")(logits)
        bounding_boxes = keras.layers.TimeDistributed(keras.layers.Dense(num_classes * 4, activation='linear'),
                                                      name='mrcnn_bbox_fc')(shared)

        def prepare_results(inputs):

            probabilities = inputs[0]
            bounding_boxes = inputs[1]

            bounding_boxes = tf.reshape(bounding_boxes, shape=(self.max_regions, self.num_classes, 4))
            class_ids = tf.argmax(probabilities, axis=1, output_type=tf.int32)
            indices = tf.stack([tf.range(probabilities.shape[0]), class_ids], axis=1)
            class_scores = tf.gather_nd(probabilities, indices)
            deltas_specific = tf.gather_nd(bounding_boxes, indices)
            class_ids = tf.to_float(class_ids)
            result = tf.concat(
                [deltas_specific, tf.expand_dims(class_ids, axis=1), tf.expand_dims(class_scores, axis=1)], axis=1)
            return result

        result = keras.layers.Lambda(lambda x: tf.map_fn(prepare_results, x, dtype=tf.float32), name='classifications')([probabilities,bounding_boxes])
        return None, result