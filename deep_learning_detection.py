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
import time
from playsound import playsound
from PIL import ImageGrab
from time import time
import socket
from goprocam import GoProCamera, constants

# IMPORTANT : this file was found on https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/?completed=/introduction-use-tensorflow-object-detection-api-tutorial/
# and updated by Lambert Rosique for PenseeArtificielle.fr for the sole purpose of a tutorial.


# ## Object detection imports
# Here are the imports from the object detection module.


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy
cap = cv2.VideoCapture(0)


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = "C:\\Users\\purva\\Documents\\model\\checkpointsMask1\\exported\\frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "C:\\Users\\purva\\Documents\\model\\map.pbtxt"
NUM_CLASSES = 2

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
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
def getSmallestBox(boxes):
    min = boxes[0][0][3] * boxes[0][0][2]
    index = 0
    for i in range(len(boxes[0][0])):
        if (boxes[0][i][3] * boxes[0][i][2]) < min:
            min = (boxes[0][i][3] * boxes[0][i][2])
            index = i
    second = 0
    for i in range(len(boxes[0][0])):
        if i is not index:
            if (boxes[0][i][3] * boxes[0][i][2]) < min:
                min = (boxes[0][i][3] * boxes[0][i][2])
                second = i
    for j in range(len(boxes[0][0])):
        if j is not index:
            boxes[0][j][0] = 0
            boxes[0][j][1] = 0
            boxes[0][j][2] = 0
            boxes[0][j][3] = 0
    return boxes[0][second]

def getSecondBig(boxes):
    max = boxes[0][0][3] * boxes[0][0][2]
    index = 0
    for i in range(len(boxes[0][0])):
        if (boxes[0][i][3] * boxes[0][i][2]) > max:
            max = (boxes[0][i][3] * boxes[0][i][2])
            index = i
    second = 0
    for i in range(len(boxes[0][0])):
        if i is not index:
            if (boxes[0][i][3] * boxes[0][i][2]) > max:
                max = (boxes[0][i][3] * boxes[0][i][2])
                second = i
    # for j in range(len(boxes[0][0])):
    #     if j is not index:
    #         boxes[0][j][0] = 0
    #         boxes[0][j][1] = 0
    #         boxes[0][j][2] = 0
    #         boxes[0][j][3] = 0
    return boxes[0][second]
def clearBoxes(boxes):
    max = boxes[0][0][3] * boxes[0][0][2]
    index = 0
    for i in range(len(boxes[0][0])):
        if (boxes[0][i][3] * boxes[0][i][2]) > max:
            max = (boxes[0][i][3] * boxes[0][i][2])
            index = i
    second = 0
    for i in range(len(boxes[0][0])):
        if i is not index:
            if (boxes[0][i][3] * boxes[0][i][2]) > max:
                max = (boxes[0][i][3] * boxes[0][i][2])
                second = i
    for j in range(len(boxes[0][0])):
        if j is not index or j is not second:
            boxes[0][j][0] = 0
            boxes[0][j][1] = 0
            boxes[0][j][2] = 0
            boxes[0][j][3] = 0
    return boxes
with detection_graph.as_default():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    # or use
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    time1 = round(time() * 1000)
    start = round(time() * 1000)
    deltas = []
    font = cv2.FONT_HERSHEY_COMPLEX

    list = os.listdir('C:\\Users\\purva\\Documents\\model\\metric_test_set_1\\')
    h = []
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # you can use other codecs as well.
    vid = cv2.VideoWriter('record.avi', fourcc, 8, (500, 490))


    gpCam = GoProCamera.GoPro()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t = time()
    gpCam.livestream("start")
    gpCam.video_settings(res='1080p', fps='30')
    gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
    cap = cv2.VideoCapture("udp://10.5.5.9:8554", cv2.CAP_FFMPEG)


    with tf.Session(graph=detection_graph, config=session_config) as sess:
        # while True:qqqq
        while True:#len(deltas) < 700
            # delta = round(time.time() * 1000) - time1
            # deltas.append(delta)
            # time1 = round(time.time() * 1000)
            ret, image_np = cap.read()
            # img = ImageGrab.grab(bbox=(100, 10, 600, 500))  # x, y, w, h
            # image_np = np.array(img)
            # frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)


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
            # TODO: restrict sound from spamming
            # if(num_detections >= 1):
            # playsound('C:\\Users\\purva\\Documents\\model\\siren.wav')
            # Visualization of the results of a detection.
            # h.append(getSecondBig(boxes))
            # boxes = clearBoxes(boxes)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=12)
            print(len(boxes.shape))
            # cv2.imwrite(
            #     'C:\\Users\\purva\\Documents\\model\\ImageResult.png',
            #     image_np)

            cv2.imshow('Deep Learning Object Detection',image_np)#, cv2.resize(image_np, (640, 480))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
            if time() - t >= 2.5:
                sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(), ("10.5.5.9", 8554))
                t = time()


        # total = 0
        # for i in deltas:
        #     total = total + i
        # average = total / len(deltas)
        # print(str(average))

# print("CONTOURS " + str(h))
a = numpy.asarray(deltas)
#numpy.savetxt('C:\\Users\\purva\\Documents\\model\\DL_Time.csv', a, delimiter=",")