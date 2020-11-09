import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import os, os.path
import cv2

# IMPORTANT : this file was found on https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/?completed=/introduction-use-tensorflow-object-detection-api-tutorial/
# and updated by Lambert Rosique for PenseeArtificielle.fr for the sole purpose of a tutorial.


# ## Object detection imports
# Here are the imports from the object detection module.


#from utils import label_map_util
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
cap = cv2.VideoCapture(0)


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = "C:\\Users\\purva\\Documents\\model\\model_dir_5\\exported\\frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', "C:\\Users\\purva\\Documents\\model\\map.pbtxt")
PATH_TO_LABELS = "C:\\Users\\purva\\Documents\\model\\map.pbtxt"
NUM_CLASSES = 1

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
    # or use
  session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
  list = os.listdir('C:\\Users\\purva\\Documents\\model\\Test_Images_5\\')

  with tf.Session(graph=detection_graph, config=session_config) as sess:
    sum = sum([len(files) for r, d, files in os.walk("C:\\Users\\purva\\Documents\\model\\Test_Images_5")])
    for i in list:
      image_np = cv2.imread("C:\\Users\\purva\\Documents\\model\\Test_Images_5\\" + str(i), cv2.IMREAD_COLOR)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=10)
      cv2.imwrite('C:\\Users\\purva\\Documents\\model\\Test_Images_5\\results\\result' + '_' + str(i) + '.jpg', image_np)
      cv2.imshow('object detection', cv2.resize(image_np, (640,480)))
      key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()