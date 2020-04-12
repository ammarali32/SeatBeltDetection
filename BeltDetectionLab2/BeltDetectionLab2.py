import cv2
from contextlib import contextmanager
import numpy


VIDEO = "test.mp4"
WEIGHTS = "YOLOFI2.weights"
CONFIG = "YOLOFI.cfg"
OBJ_NAMES = "obj.names"


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def get_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[layer[0] - 1] for layer in net.getUnconnectedOutLayers()]


def get_classes():
    classes = []
    with open(OBJ_NAMES, "r") as file:
        classes = [line.strip() for line in file.readlines()]
    return classes


def belt_detector(net, img):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    height, width, channels = img.shape

    belt_corner_detected = False
    belt_detected = False

    outs = net.forward(get_layers(net))
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = numpy.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if class_id == 1:
                    belt_corner_detected = True
                elif class_id == 0:
                    belt_detected = True
    return belt_detected, belt_corner_detected


def main():
    with video_capture(VIDEO) as cap:
        net = cv2.dnn.readNet(WEIGHTS, CONFIG)

        successful_detections = 0
        frames = 0
        while True:
            frame = cap.read()
            if not frame[0]:
                break
            img = frame[1]

            # TODO: your code here

            belt_detected, belt_corner_detected = belt_detector(net, img)
            frames += 1
            if belt_detected:
                successful_detections += 1
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == 27:
                break
        print("Total frames {}, successful detections {}".format(frames, successful_detections))


if __name__ == "__main__":
    main()
