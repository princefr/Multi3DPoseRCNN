from tensorflow.python import keras
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.layers import Layer as KL
from tensorflow.python.keras import engine as KE
import numpy as np
import math

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KL):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates
    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self,
                 image_shape=None,
                 max_proposals=None,
                 pre_nms_max_proposals=None,
                 bounding_box_std_dev=None,
                 nms_threshold=None,
                 anchor_scales=None,
                 anchor_ratios=None,
                 backbone_strides=None,
                 anchor_stride=None,
                 **kwargs):

        super(ProposalLayer, self).__init__(**kwargs)

        self.image_shape = image_shape
        self.max_proposals = max_proposals
        self.pre_nms_max_proposals = pre_nms_max_proposals
        self.bounding_box_std_dev = bounding_box_std_dev
        self.nms_threshold = nms_threshold
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.backbone_strides = backbone_strides
        self.anchor_stride = anchor_stride
        assert max_proposals != None

        backbone_shapes = np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in backbone_strides])

        self.anchors = norm_boxes(generate_pyramid_anchors(anchor_scales,
                                                anchor_ratios,
                                                backbone_shapes,
                                                backbone_strides,
                                                anchor_stride), image_shape)

    def get_config(self):
        config = super(ProposalLayer, self).get_config()
        config['image_shape'] = self.image_shape
        config['max_proposals'] = self.max_proposals
        config['pre_nms_max_proposals'] = self.pre_nms_max_proposals
        config['bounding_box_std_dev'] = self.bounding_box_std_dev
        config['nms_threshold'] = self.nms_threshold
        config['anchor_scales'] = self.anchor_scales
        config['anchor_ratios'] = self.anchor_ratios
        config['backbone_strides'] = self.backbone_strides
        config['anchor_stride'] = self.anchor_stride
        return config

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.bounding_box_std_dev, [1, 1, 4])
        # Anchors
        anchors = self.anchors
        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.pre_nms_max_proposals, tf.shape(anchors)[0])

        scores, ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors")

        def gather(inputs):
            return tf.gather(inputs[0], inputs[1])

        deltas = tf.map_fn(gather,[deltas, ix], dtype=deltas.dtype)

        def gather_anchors(index):
            return tf.gather(anchors,index)

        pre_nms_anchors = tf.map_fn(gather_anchors,ix, dtype=anchors.dtype)

        def apply_box_deltas(inputs):
            return apply_box_deltas_graph(inputs[0], inputs[1])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = tf.map_fn(apply_box_deltas,[pre_nms_anchors, deltas], dtype=deltas.dtype)

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)

        def clip_boxes(boxes):
            return clip_boxes_graph(boxes, window)
        boxes = tf.map_fn(clip_boxes,boxes, dtype=boxes.dtype)

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(inputs):
            boxes = inputs[0]
            scores = inputs[1]
            indices = tf.image.non_max_suppression(
                boxes, scores, self.max_proposals,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.max_proposals - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = tf.map_fn(nms,[boxes, scores], dtype=boxes.dtype)
        proposals = keras.layers.Reshape((self.max_proposals,4), name="proposals")(proposals)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.max_proposals, 4)

def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)