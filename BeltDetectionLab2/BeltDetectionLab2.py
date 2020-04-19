import math

import numpy as np
import cv2
import os
import time
import numpy as np
from imutils import face_utils
import imutils


def main():
    net = cv2.dnn.readNet("YOLOFI2.weights", "YOLOFI.cfg")
    cap = cv2.VideoCapture("test.mp4")
    classes = []
    l = 1
    with open("obj.names", "r")as f:
        classes = [line.strip() for line in f.readlines()]
        layers_names = net.getLayerNames()
        outputlayers = [layers_names[i[0] - 1]
                        for i in net.getUnconnectedOutLayers()]
        frame_id = 0
        err = 0

        while True:
            _, frame = cap.read()
            frame_id += 1
            belt_corner_detected = False
            belt_detected = False
            height, width, channels = frame.shape

            # Type you code here

            # frame = apply_sharpening(frame)
            # frame = apply_threshold(frame)
            # frame = apply_histogram_equalization(frame)

            frame = apply_clahe(frame)
            # frame = apply_cleaning(frame)

            blob = cv2.dnn.blobFromImage(
                frame, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(outputlayers)

            confidence = 0
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.2:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        cv2.rectangle(
                            frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if class_id == 1:
                            belt_corner_detected = True
                        elif class_id == 0:
                            belt_detected = True

            print(belt_detected)
            cv2.imshow("Image", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# Less effective then clahe
def apply_histogram_equalization(frame):
    R, G, B = cv2.split(frame)

    eh_R = cv2.equalizeHist(R)
    eh_G = cv2.equalizeHist(G)
    eh_B = cv2.equalizeHist(B)

    res = cv2.merge((eh_R, eh_G, eh_B))

    return res


# Yeah, by increasing tile grid size from 8 to 16 i got
# just one false-negative
def apply_clahe(frame):
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16))

    R, G, B = cv2.split(frame)

    clahe_R = clahe.apply(R)
    clahe_G = clahe.apply(G)
    clahe_B = clahe.apply(B)

    res = cv2.merge((clahe_R, clahe_G, clahe_B))

    return res


# Seems to be a heavy task
# After clahe setup applying this gives less accuracy
def apply_cleaning(frame):
    h = 15
    template_window_size = 7
    search_window_size = 21
    clean = cv2.fastNlMeansDenoising(
        frame, None, h, template_window_size, search_window_size)

    return clean


# Useless :(
def apply_sharpening(frame):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    res = cv2.filter2D(frame, -1, kernel)

    return res


# Seems cool & in most of frames edges are clear, probably
# if combine this output with something else it's possible to
# achieve good results. But not in this lab)
def apply_threshold(frame):
    R, G, B = cv2.split(frame)

    R = cv2.adaptiveThreshold(
        R, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    G = cv2.adaptiveThreshold(
        G, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    B = cv2.adaptiveThreshold(
        B, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    res = cv2.merge((R, G, B))

    return res


if __name__ == '__main__':
    main()
