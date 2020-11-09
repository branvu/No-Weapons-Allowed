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

import matplotlib.pyplot as plt

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
PATH_TO_CKPT = "C:\\Users\\purva\\Documents\\model\\model_dir_5\\exported\\frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
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
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def CVErrorDL(CVcontours,Dlcontours):
    err=[]
    returnArray=[]
    for i in range(len(CVcontours)):
        M = cv2.moments(CVcontours[i])
        cv= [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        tuple=[0,0]
        tuple[0]=i
        for j in range(len(Dlcontours)):
            tuple[1]=j
            dl=[((Dlcontours[0][j][0] + Dlcontours[0][j][3]) / 2), ((Dlcontours[0][j][1] + Dlcontours[0][j][2]) / 2)]
            err.append(getDistance(cv,dl))
            returnArray.append(tuple)
    min=0
    underFifty=0
    found = False
    for k in range(len(err)):
        if err[k]<180:

            found = True
            underFifty=k
            break
        # print(err[k])

    if found is False:
        return False,CVcontours[0],Dlcontours[0]
    else:
        returnCV=returnArray[underFifty][0]
        returnDL=returnArray[underFifty][1]
        return True,returnCV,Dlcontours[returnDL]

def greatest_contour(contours):
    max = cv2.contourArea(contours[0])
    index = 0
    max_index = 0
    for c in contours:
        if cv2.contourArea(c) > max:
            max = cv2.contourArea(c)
            max_index = index
        index = index + 1
    return contours[max_index]


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def getDistance(center1, center2):
    x = (center1[0] - center2[0]) * (center1[0] - center2[0])
    y = (center1[1] - center2[1]) * (center1[1] - center2[1])
    return (x + y)**0.5

def getDistanceBoxClosestToPoint(boxes, center):
    min = getDistance([((boxes[0][0][0] + boxes[0][0][3]) / 2), ((boxes[0][0][1] + boxes[0][0][2]) / 2)], center)
    for i in range(len(boxes[0][0])):
        if getDistance([(boxes[0][i][0] + boxes[0][i][3]) / 2, ((boxes[0][i][1] + boxes[0][i][2]) / 2)],
                       center) < min:
            min = getDistance([((boxes[0][i][0] + boxes[0][i][3]) / 2), ((boxes[0][i][1] + boxes[0][i][2]) / 2)],
                              center)
    return min

def getBoxClosestToPoint(boxes, center):
    min = getDistance([int((boxes[0][0][0] + boxes[0][0][3]) / 2), int((boxes[0][0][1] + boxes[0][0][2]) / 2)], center)
    index = 0
    for i in range(len(boxes[0][0])):
        if getDistance([int((boxes[0][i][0] + boxes[0][i][3]) / 2), int((boxes[0][i][1] + boxes[0][i][2]) / 2)], center) < min:
            min = getDistance([int((boxes[0][i][0] + boxes[0][i][3]) / 2), int((boxes[0][i][1] + boxes[0][i][2]) / 2)], center)
            index = i
    return boxes[0][index]
with detection_graph.as_default():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    # or use
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    time1 = round(time.time() * 1000)
    start = round(time.time() * 1000)
    deltas = []
    timeDeep=time.time()*1000
    first = True
    font = cv2.FONT_HERSHEY_COMPLEX
    cv_check = False
    text= "0"
    center_contour = 0
    list = os.listdir('C:\\Users\\purva\\Documents\\model\\metric_test_set_1\\Data\\')

    with tf.Session(graph=detection_graph, config=session_config) as sess:
        # while True:
        while True:
        # while len(deltas) <= 700:#


            delta = round(time.time() * 1000) - time1

            deltas.append(delta)
            time1 = round(time.time() * 1000)

            ret, image_np = cap.read()
            write_cv = True
            # raw = image_np.copy()
            #
            # image_np = cv2.imread("C:\\Users\\purva\\Documents\\model\\metric_test_set_1\\Data\\" + str(i), cv2.IMREAD_UNCHANGED)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
            rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            lower_hsv = np.array([21, 113, 0])
            upper_hsv = np.array([183, 118, 77])
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            mask_black = cv2.inRange(rgb, np.array([0, 0, 0]), np.array([30, 50, 35]))
            new = mask + mask_black
            mask2 = cv2.blur(new, (4, 4))
            kernel = np.ones((5, 5), np.uint8)
            kernel_dilate = np.ones((2, 2), np.uint8)
            dilation = cv2.dilate(mask2, kernel_dilate, iterations=1)
            closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
            mask1 = cv2.GaussianBlur(closing, (9, 9), 0)
            contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            approx = None

            for cnt in contours:
                area = cv2.contourArea(cnt)
                M = cv2.moments(greatest_contour(contours))
                center_contour = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]

                approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
                x = approx.ravel()[0]
                y = approx.ravel()[1]
                if area > 2000 and area < 10000:
                    if 3 <= len(approx) <= 6:
                        cv_check = True
                        # cv2.drawContours(image_np, [approx], 0, (0, 0, 0), 5)

                        break
                    else:
                        cv_check = False


                else:
                    cv_check = False


            if cv_check is True or first is True:
                first = False

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
                # print(boxes[0][0])

                # center_1 = [int((boxes[0][0][0] + boxes[0][0][3]) / 2), int((boxes[0][0][1] + boxes[0][0][2]) / 2)]
                # center_2 = [int((boxes[0][1][0] + boxes[0][1][3]) / 2), int((boxes[0][1][1] + boxes[0][1][2]) / 2)]

                found,contour,box=CVErrorDL(contours,boxes)
                box1 = [[box]]
                print(found)
                if found:
                    write_cv = False
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(box1),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=12)
                    text = "Mixed Algorithm"

                else:
                    cv_check = False


                    #print("Confirmed detection at: " + str(center_contour))

                # elif write_cv is True:
                #     print("(ONLY CV)")
                #     for cnt in contours:
                #         area = cv2.contourArea(cnt)
                #         M = cv2.moments(greatest_contour(contours))
                #         center_contour = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
                #
                #         approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
                #         x = approx.ravel()[0]
                #         y = approx.ravel()[1]
                #
                #         if area > 2000 and area < 12000:
                #             if 3 <= len(approx) <= 10:
                #                 cv_check = True
                #                 cv2.drawContours(image_np, [approx], 0, (0, 0, 0), 5)
                #                 cv2.putText(image_np, "Weapon", (x, y), font, 0.5, (0, 0, 0))
                #             else:
                #                 cv_check = False


            if cv_check is False and (time.time()*1000-timeDeep)>300:
                text = "Deep Learning"
                timeDeep = time.time()*1000
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

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
                # boxes = clearBoxes(boxes)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=12)

            cv2.putText(image_np, text, (0, 350), font, 1, (0, 0, 255))
            # cv2.imwrite(
            #     'C:\\Users\\purva\\Documents\\model\\mixed_4_results\\' + '_' + str(i) + '.jpg',
            #
            #     image_np)

            cv2.imshow('Mixed Detection', cv2.resize(image_np, (640, 480)))


            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        total = 0
        # a = numpy.asarray(deltas)
        # numpy.savetxt('C:\\Users\\purva\\Documents\\model\\deltas_mixed_detection_v1.csv', a, delimiter=",")
        for i in deltas:
            total = total + i
        # average = total / len(deltas)
        # (str(average))
