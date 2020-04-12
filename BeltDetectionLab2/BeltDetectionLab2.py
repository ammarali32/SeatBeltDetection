import cv2
from contextlib import contextmanager
import numpy as np
import logging


VIDEO = "test.mp4"
WEIGHTS = "YOLOFI2.weights"
CONFIG = "YOLOFI.cfg"
OBJ_NAMES = "obj.names"


logging.basicConfig(level=logging.INFO)


class BeltVisible:
    def __init__(self, belt_frames, belt_corner_frames):
        self.belt_frames = belt_frames
        self.belt_corner_frames = belt_corner_frames


class BeltDetected:
    def __init__(self):
        self.belt_frames = []
        self.belt_corner_frames = []

    def add_belt(self, frame):
        self.belt_frames.append(frame)

    def add_corner_belt(self, frame):
        self.belt_corner_frames.append(frame)


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


def belt_detector(net, img, belt_detected, current_frame):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (480, 480), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    height, width, channels = img.shape

    outs = net.forward(get_layers(net))
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
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if class_id == 1:
                    belt_detected.add_corner_belt(current_frame)
                elif class_id == 0:
                    belt_detected.add_belt(current_frame)

    return belt_detected


def print_belt_report(belt_detected, total_frames):
    belt_visible = BeltVisible(
        belt_frames=[i for i in range(124)],
        belt_corner_frames=[i for i in range(124)]
    )
    success_belt_frames = set(belt_visible.belt_frames).intersection(belt_detected.belt_frames)
    success_belt_corner_frames = set(belt_visible.belt_corner_frames).intersection(
        belt_detected.belt_corner_frames
    )

    logging.info(
        "Total frames {}, successfully detected belt {} of {} times, corner belt - {} of {}".format(
            total_frames,
            len(success_belt_frames),
            len(belt_visible.belt_frames),
            len(success_belt_corner_frames),
            len(belt_visible.belt_corner_frames)
        ))
    logging.info("Non detected belt frames: {}".format(
        set(belt_visible.belt_frames).difference(belt_detected.belt_frames))
    )
    logging.info("False detected belt frames: {}".format(
        set(belt_detected.belt_frames).difference(belt_visible.belt_frames))
    )
    logging.info("Non detected belt corner frames: {}".format(
        set(belt_visible.belt_corner_frames).difference(belt_detected.belt_corner_frames))
    )
    logging.info("False detected belt corner frames: {}".format(
        set(belt_detected.belt_corner_frames).difference(belt_visible.belt_corner_frames))
    )


def main():
    with video_capture(VIDEO) as cap:
        net = cv2.dnn.readNet(WEIGHTS, CONFIG)
        frame_id = 0
        belt_detected = BeltDetected()
        while True:
            frame = cap.read()
            frame_id += 1
            if not frame[0]:
                break
            img = frame[1]

            # TODO: your code here

            belt_detected = belt_detector(net, img, belt_detected, frame_id)
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == 27:
                break
        print_belt_report(belt_detected, frame_id + 1)


if __name__ == "__main__":
    main()
