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
import numpy

# IMPORTANT : this file was found on https://pythonprogramming.net/video-tensorflow-object-detection-api-tutorial/?completed=/introduction-use-tensorflow-object-detection-api-tutorial/
# and updated by Lambert Rosique for PenseeArtificielle.fr for the sole purpose of a tutorial.
test_metric = False
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

cap = cv2.VideoCapture(0)
time1 = round(time.time() * 1000)
start = round(time.time() * 1000)
deltas = []
font = cv2.FONT_HERSHEY_COMPLEX
list = os.listdir('C:\\Users\\purva\\Documents\\model\\metric_test_set_1\\')
h = []
# for i in list:
while True:#len(deltas) < 700
    delta = round(time.time() * 1000) - time1
    deltas.append(delta)
    time1 = round(time.time() * 1000)
    # if test_metric:
    #     #image_np = cv2.imread("C:\\Users\\purva\\Documents\\model\\metric_test_set_1\\" + str(i), cv2.IMREAD_UNCHANGED)
    # else:
    ret, image_np = cap.read()

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

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if area > 2000 and area < 12000:
            if 3 <= len(approx) <= 10:
                con = greatest_contour(contours)
                cv2.drawContours(image_np, [approx], 0, (0, 0, 0), 5)
                cv2.putText(image_np, "Weapon", (x, y), font, 0.5, (0, 0, 0))
    if test_metric:
        # cv2.imwrite('C:\\Users\\purva\\Documents\\model\\metric_test_set_2\\results_cv_only\\result' + '_' + str(i) + '.jpg', image_np)
        M = cv2.moments(greatest_contour(contours))
        center_contour = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        h.append(center_contour)
    cv2.imshow('df',mask2)
    cv2.imshow('Computer Vision Detection Only', cv2.resize(image_np, (640, 480)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

total = 0
for i in deltas:
    total = total + i
average = total / len(deltas)
print("Average detection time: " + str(average))
# print("CONTOURS " + str(h))
# a = numpy.asarray(deltas)
# numpy.savetxt('C:\\Users\\purva\\Documents\\model\\CVTIME.csv', a, delimiter=",")
