
import cv2
import numpy as np
cap = cv2.VideoCapture("136_2.mp4")
object_detection = cv2.createBackgroundSubtractorMOG2()


def empty(x):
    pass


cv2.namedWindow("Trackbars")
cv2.createTrackbar("Hue min", "Trackbars", 80, 255, empty)
cv2.createTrackbar("Hue max", "Trackbars", 150, 255, empty)
cv2.createTrackbar("Sut min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Sut max", "Trackbars", 230, 255, empty)
cv2.createTrackbar("Val min", "Trackbars", 80, 255, empty)
cv2.createTrackbar("Val max", "Trackbars", 255, 255, empty)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    if not ret:
        break

    width, height, _ = frame.shape

    roi = frame[200:520, 380:980]

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sut min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sut max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = object_detection.apply(roi)

    mask = cv2.inRange(hsv_frame, lower, upper)

    constours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in constours:
        area = cv2.contourArea(cnt)
        if area > 100 and area < 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            coord_x_center = int(x + w/2)
            coord_y_center = int(y + h/2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.line(frame, (coord_x_center, 0),
                     (coord_x_center, 1280), (0, 0, 255), 1)
            cv2.line(frame, (0, coord_y_center),
                     (1280, coord_y_center), (0, 0, 255), 1)

        # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('mask', mask)

    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
