import matplotlib
import matplotlib.pyplot as plt

import os
import cv2
import glob

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--export_model_path", type=str, required=True, help="path to export model")
ap.add_argument("-l", "--labelmap_path", type=str, required=True, help="path to label map")
ap.add_argument("-i", "--input_image_path", type=str, required=True, help="path to input image")
ap.add_argument("-o", "--output_image_path", type=str, help="path to output image")
args = vars(ap.parse_args())

"""
python inference_image.py -p custom_trained_models/efficientdet_d0_coco17_tpu-32/export -l 'dataset/labelmap.pbtxt' -i dataset/test_images/test_1.jpg -o dataset/test_images_output/test_1.jpg
"""

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: the file path to the image

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


# Load the configurations, initialize paths and resotore the export checkpoint

MODEL_TEST = args["export_model_path"] #"./custom_trained_models/efficientdet_d0_coco17_tpu-32/export"

pipeline_config = os.path.join(MODEL_TEST, 'pipeline.config')
model_dir_test = os.path.join(MODEL_TEST, 'checkpoint/ckpt-0')
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
            model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(
            model=detection_model)
ckpt.restore(os.path.join(model_dir_test))


# load labelmap, map them to it's labels and load the detector

detect_fn = get_model_detection_function(detection_model)

#map labels for inference decoding
label_map = label_map_util.load_labelmap(args["labelmap_path"])  #'dataset/labelmap.pbtxt'
categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)


image_np = load_image_into_numpy_array(args["input_image_path"])
    
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.6,
            agnostic_mode=False
)

# plt.imshow(image_np_with_detections)
cv2.imwrite(args["output_image_path"], cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))