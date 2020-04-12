import os
import time

import numpy as np
import cv2


def main():
    net = cv2.dnn.readNet("YOLOFI2.weights", "YOLOFI.cfg")
    cap = cv2.VideoCapture("test.mp4")

    classes = []
    l = 1

    with open("obj.names", "r")as f:
        classes = [line.strip() for line in f.readlines()]
        count_classes = len(classes)
        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(count_classes, 3))
        font = cv2.FONT_HERSHEY_PLAIN
        frame_id = 0
        dd = -1
        time_now = time.time()
        frame_id = 0
        err = 0
        while True:
            _, frame = cap.read()

            if frame is None:
                break

            frame_id += 1
            beltcornerdetected = False
            beltdetected = False
            height, width, channels = frame.shape

            # Type you code here

            new_frame = frame.copy()

            clahe = cv2.createCLAHE(clipLimit=200.0, tileGridSize=(8, 8))
            new_frame[:, :, 0] = clahe.apply(frame[:, :, 0])
            new_frame[:, :, 1] = clahe.apply(frame[:, :, 1])
            new_frame[:, :, 2] = clahe.apply(frame[:, :, 2])

            kern = cv2.getGaborKernel((10, 10), 2, np.pi / 16 * 15, 5, 50, 1, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()

            new_frame = cv2.filter2D(new_frame, cv2.CV_8UC3, kern)

            new_frame = cv2.fastNlMeansDenoising(new_frame, None, 10, 30, 1)

            blob = cv2.dnn.blobFromImage(new_frame, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
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

            print(f'{frame_id}: {beltdetected}')

            cv2.imshow("Image", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
