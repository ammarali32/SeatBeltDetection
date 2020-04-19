import numpy as np
import cv2
import os
import time
import numpy as np
from imutils import face_utils
import imutils

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

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

        pred_corrects = []
        while True:
            _, frame = cap.read()
            frame_id += 1
            beltcornerdetected = False
            beltdetected = False
            if frame is None:
                break
            height, width, channels = frame.shape

            # # Type you code here
            # orig_frame = frame
            # frame = cv2.resize(frame, (300, 300))
            # frame = frame[136:356, 115:335]
            # frame = cv2.resize(frame, (480, 480))
            frame = increase_brightness(frame, 10)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame = cv2.fastNlMeansDenoising(frame, h=1, templateWindowSize=7, searchWindowSize=21) 
            frame = cv2.bilateralFilter(frame, d=-1, sigmaSpace=20, sigmaColor=5)
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
            frame = clahe.apply(frame)
            # frame = cv2.equalizeHist(frame)
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            

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

            # output whether prediction is correct
            print(frame_id)
            is_correct = 0
            if beltdetected and frame_id <= 125:
                is_correct = 1
            elif not beltdetected and frame_id > 125:
                is_correct = 1
            print(is_correct)
            pred_corrects.append(is_correct)
            # print(beltdetected)
            # cv2.imshow("Image", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            print("Accuracy: {}".format(sum(pred_corrects) / len(pred_corrects)))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()