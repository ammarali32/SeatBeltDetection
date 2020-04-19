import cv2
import numpy as np


def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 0.3, theta, 9.0, 0.6, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def main():
    net = cv2.dnn.readNet("YOLOFI2.weights", "YOLOFI.cfg")
    cap = cv2.VideoCapture("test.mp4")
    with open("obj.names", "r")as f:
        layers_names = net.getLayerNames()
        outputlayers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        count = 0  # for counting frames

        while True:
            _, frame = cap.read()
            if frame is None:
                break
            beltdetected = False
            height, width, channels = frame.shape
            frame = cv2.addWeighted(frame, 1.3, frame, 0,-10)  # change the contrast of the frame

            filters = build_filters()
            frame = process(frame, filters)

            frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 10, 7, 30)  # also use the NON-LOCAL MEANS DENOISING ALGOROTHM
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(outputlayers)

            for out in outs:

                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence != 0:
                        print(confidence)  # to find optimal one
                    if confidence > 0.2:  # decrease confidence cos it is accurate enought
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if class_id == 0:
                            beltdetected = True

            print(count, ' ', beltdetected)
            count += 1
            cv2.imshow("Image", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
