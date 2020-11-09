import cv2
import numpy as np



def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
# cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
# cv2.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
# cv2.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
# cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
# cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)
LR = 0
LG = 0
LB = 0
UR = 0
UG = 0
UB = 0
cv2.createTrackbar("LR", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("LG", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("LB", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("UR", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("UG", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("UB", "Trackbars", 0, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #0,0,153
    #180, 46, 247

    LR = cv2.getTrackbarPos("LH", "Trackbars")
    LG = cv2.getTrackbarPos("LS", "Trackbars")
    LB = cv2.getTrackbarPos("LV", "Trackbars")
    UR = cv2.getTrackbarPos("UH", "Trackbars")
    UG = cv2.getTrackbarPos("US", "Trackbars")
    UB = cv2.getTrackbarPos("UV", "Trackbars")

    lower = np.array([LR,LG,LB])
    higher = np.array([UR, UG, UB])
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([180, 255, 113])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # mask = cv2.inRange(rgb, lower, higher)
    #0,0,0
    #180, 255, 113
    # Contours detection

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        i = i + 1
        if area > 1000 and area < 15000:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
            # cv2.fillPoly(frame, pts=[approx], color=(255, 255, 255))
            # print(cv2.contourArea(cnt))
            cv2.putText(frame, "Gun", (x, y), font, 1, (0, 0, 0))


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()