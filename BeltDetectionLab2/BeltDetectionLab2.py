import cv2
import os
import time
import numpy as np
from imutils import face_utils
import imutils
from matplotlib import pyplot as plt


def cl(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def build_filters(filters):
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((31, 31), 2.3, theta, 12.0, 70.0, 2, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)

    return filters


def process(image, filters):
    accum = np.zeros_like(image)
    for kern in filters:
        fimg = cv2.filter2D(image, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)

    return accum


def main():
    net = cv2.dnn.readNet("YOLOFI2.weights", "YOLOFI.cfg")
    cap = cv2.VideoCapture("test.mp4")
    classes = []
    l = 1
    with open("obj.names", "r")as f:
        classes = [line.strip() for line in f.readlines()]
        layers_names = net.getLayerNames()
        outputlayers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        font = cv2.FONT_HERSHEY_PLAIN
        frame_id = 0
        dd = -1
        time_now = time.time()
        frame_id = 0
        err = 0

        counter = 1

        while True:
            _, frame = cap.read()
            frame_id += 1
            beltcornerdetected = False
            beltdetected = False
            height, width, channels = frame.shape

            # Type you code here

            # Denoising did not help :(
            # frame = cv2.fastNlMeansDenoising(frame, None, 10, 7, 21)
            #
            # b, g, r = cv2.split(frame)  # get b,g,r
            # frame = cv2.merge([r, g, b])  # switch it to rgb

            # CLAHE https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
            frame = cl(frame)

            # GABOR FILTERS https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
            filters = []
            filters = build_filters(filters)
            frame = process(frame, filters)

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(outputlayers)
            class_ids = []
            boxes = []
            shape = []
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
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if class_id == 1:
                            beltcornerdetected = True
                        elif class_id == 0:
                            beltdetected = True

            print(counter, ' ', beltdetected)
            counter += 1
            cv2.imshow("Image", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

