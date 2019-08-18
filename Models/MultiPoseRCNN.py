import tensorflow as tf
from tensorflow.python import keras
import sys
sys.path.append("../")
from Layers.Detection import DetectionLayer
from Layers.FPNClassifier import FPNClassifier
from Layers.FPNMask import FPNMask
from Layers.Proposal import ProposalLayer
from Layers.RPN import RPN
from Layers.RootNet import RootNet
from Layers.PoseNet import PoseNet
from Utils.Config import Config
from Models.Backbone import Backbone


"""
person search
https://arxiv.org/pdf/1604.01850.pdf
https://github.com/ShuangLI59/person_search
"""

"""
this directory is inspired from
https://github.com/edouardlp/Mask-RCNN-Keras
"""



class MultiPoseRCNN():
    def __init__(self, config, model_dir=None, initial_keras_weights=None, mode="Train"):
        assert mode in ['Train', 'Inference']
        self.config = config
        self.mode = mode

    def Resnet_direct(self):
        model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False)
        return keras.Model(model.input, model.output)

    def _build_pose_net(self):
        input_image = keras.layers.Input(shape=self.config.input_image_shape, name="input_image")
        resnet = self.Resnet_direct()
        pose_net = PoseNet(rois=None, feature_maps=resnet(input_image), pool_size=None, image_shape=None, num_classes=None, max_regions=None, fc_layers_size=None,
                 pyramid_top_down_size=None, joint_num=self.config.joint_num, depth_dim=self.config.depth_dim, config=self.config)

        return pose_net.build_training()


    def _build_root_net(self):
        input_image = keras.layers.Input(shape=self.config.input_image_shape, name="input_image")
        k_value_input = keras.layers.Input(shape=(None, None, None))
        resnet = self.Resnet_direct()
        root_net = RootNet(rois=None,
                 feature_maps=resnet(input_image),
                 pool_size=None,
                 image_shape=None,
                 num_classes=None,
                 max_regions=None,
                 fc_layers_size=None,
                 pyramid_top_down_size=None, joint_num=self.config.joint_num, depth_dim = self.config.depth_dim, config=self.config, k_value=k_value_input)

        return root_net.build_training()


    def _build_keras_models(self, config, mode="Train"):
        print("loading models")
        input_image = keras.layers.Input(shape=config.input_image_shape, name="input_image")
        k_value_input = keras.layers.Input(shape=(None, None, None))

        if mode == "Train":
            input_id = keras.layers.Input(shape=[1], name="input_id", dtype=tf.string)
            input_bounding_box = keras.layers.Input(shape=[4], name="input_image_bounding_box", dtype=tf.float32)
            input_original_shape = keras.layers.Input(shape=[3], name="input_original_shape", dtype=tf.int32)


        backbone = Backbone(input_tensor=input_image, architecture=config.architecture, pyramid_size=config.pyramid_top_down_size)
        P2, P3, P4, P5, P6 = backbone.build()
        rpn = RPN(anchor_stride=config.anchor_stride, anchors_per_location=config.anchors_per_location, depth=config.pyramid_top_down_size, feature_maps=[P2, P3, P4, P5, P6])

        # anchor_object_probs: Probability of each anchor containing only background or objects
        # anchor_deltas: Bounding box refinements to apply to each anchor to better enclose its object
        anchor_object_probs, anchor_deltas = rpn.build()

        # rois: Regions of interest (regions of the image that probably contain an object)
        proposal_layer = ProposalLayer(name="ROI", image_shape=config.input_image_shape[0:2], max_proposals=config.max_proposals, pre_nms_max_proposals=config.pre_nms_max_proposals, bounding_box_std_dev=config.bounding_box_std_dev,
                                       nms_threshold=config.proposal_nms_threshold,
                                       anchor_scales=config.anchor_scales,
                                       anchor_ratios=config.anchor_ratios,
                                       backbone_strides=config.backbone_strides,
                                       anchor_stride=config.anchor_stride)

        rois = proposal_layer([anchor_object_probs, anchor_deltas])
        mrcnn_feature_maps = [P2, P3, P4, P5]


        # fpn classifier
        fpn_classifier_graph = FPNClassifier(rois=rois, feature_maps=mrcnn_feature_maps,
                                                  pool_size=config.classifier_pool_size,
                                                  image_shape=config.input_image_shape,
                                                  num_classes=config.num_classes,
                                                  max_regions=config.max_proposals,
                                                  fc_layers_size=config.fc_layers_size,
                                                  pyramid_top_down_size=config.pyramid_top_down_size)

        # rois_class_probs: Probability of each class being contained within the roi
        # classifications: Bounding box refinements to apply to each roi to better enclose its object
        fpn_classifier_model, classifications = fpn_classifier_graph.build()

        detection_inputs = [rois, classifications]

        if mode == "Train":
            detection_inputs.append(input_bounding_box)

        detections = DetectionLayer(name="detections",
                                    max_detections=config.max_detections,
                                    bounding_box_std_dev=config.bounding_box_std_dev,
                                    detection_min_confidence=config.detection_min_confidence,
                                    detection_nms_threshold=config.detection_nms_threshold,
                                    image_shape=config.input_image_shape)(detection_inputs)

       # fpn_mask_graph = FPNMask(rois=detections, feature_maps=mrcnn_feature_maps, pool_size=config.mask_pool_size, image_shape=config.input_image_shape[0:2], num_classes=config.num_classes, max_regions=config.max_detections,pyramid_top_down_size=config.pyramid_top_down_size)

        #fpn_mask_model, masks = fpn_mask_graph.build()




        # build rootnet
        rootnet_graph = RootNet(rois=detections,
                                      feature_maps=mrcnn_feature_maps,
                                      pool_size=config.mask_pool_size,
                                      image_shape=config.input_image_shape[0:2],
                                      num_classes=config.num_classes,
                                      max_regions=config.max_detections,
                                      pyramid_top_down_size=config.pyramid_top_down_size, joint_num=self.config.joint_num, depth_dim=self.config.depth_dim, config=self.config, k_value=k_value_input)

        coord_out_rootnet, model_rootnet = rootnet_graph.build()

        # build posenet
        posenet_graph = PoseNet(rois=detections,
                                      feature_maps=mrcnn_feature_maps,
                                      pool_size=config.mask_pool_size,
                                      image_shape=config.input_image_shape[0:2],
                                      num_classes=config.num_classes,
                                      max_regions=config.max_detections,
                                      pyramid_top_down_size=config.pyramid_top_down_size, joint_num=self.config.joint_num, depth_dim=self.config.depth_dim, config=self.config)

        coord_out_posenet, model_posenet = posenet_graph.build()






        # defining input and outputs
        inputs = [input_image]
        outputs = []

        if mode == "Train":
            inputs.extend([input_id, input_bounding_box, input_original_shape])
            outputs.extend([input_id, input_bounding_box, input_original_shape])

        outputs.extend([detections, coord_out_posenet, coord_out_rootnet])

        multipose_rcnn_model = keras.models.Model(inputs, outputs, name='multipose_rcnn_model')
        return multipose_rcnn_model, fpn_classifier_model, model_posenet, model_rootnet, proposal_layer.anchors

    def train_pose_net(self, train_generator, validation_generator, save_model=True):
        print("PoseNet training is starting")
        model = self._build_pose_net()
        print("printing posenet model summary")
        model.summary()
        print("compiling the posenet model")
        model.compile(optimizer=keras.optimizers.Adam(self.config.lr), loss="")
        print("training posenet")
        model.fit_generator(train_generator, epochs=100, validation_data=validation_generator, validation_steps=100)
        if save_model:
            model.save("pose_net.h5")


    def train_root_net(self, train_generator, validation_generator, save_model=True):
        print("RootNet training is starting")
        model = self._build_pose_net()
        print("printing rootnet model summary")
        model.summary()
        print("compiling the root_net model")
        model.compile(optimizer=keras.optimizers.Adam(self.config.lr), loss="")
        print("training rootnet")
        model.fit_generator(train_generator, epochs=100, validation_data=validation_generator, validation_steps=100)
        if save_model:
            model.save("root_net.h5")


    def get_trained_keras_models(self):
        multipose_rcnn_model, fpn_classifier_model,  posenet_model, rootnet_model, anchors = self._build_keras_models()
        multipose_rcnn_model.load_weights(self.initial_keras_weights, by_name=True)
        fpn_classifier_model.load_weights(self.initial_keras_weights, by_name=True)
        posenet_model.load_weights(self.initial_keras_weights, by_name=True)
        rootnet_model.load_weights(self.initial_keras_weights, by_name=True)
        return multipose_rcnn_model, fpn_classifier_model,  anchors, posenet_model, rootnet_model