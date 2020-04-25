
from os.path import dirname, abspath

import cv2
from contextlib import contextmanager
import numpy as np
import logging


VIDEO = "test.mp4"
WEIGHTS = "YOLOFI2.weights"
CONFIG = "YOLOFI.cfg"
OBJ_NAMES = "obj.names"
SAVE_PATH = dirname(dirname(abspath(__file__))) + "/"


logging.basicConfig(level=logging.INFO)


class BeltVisible:
    # lists of frames (ids) where the belt part is supposed to be detected as closed
    # first frame has id 0
    def __init__(self, belt_frames, belt_corner_frames):
        self.belt_frames = belt_frames
        self.belt_corner_frames = belt_corner_frames


class BeltDetected:
    # list of frames (ids) where the belt part was detected as closed
    # first frame has id 0
    def __init__(self):
        self.belt_frames = []  # main part
        self.belt_corner_frames = []  # corner part

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
        belt_frames=[i for i in range(125)],
        belt_corner_frames=[i for i in range(125)]
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


def apply_clahe(img, **kwargs):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab = cv2.split(lab)
    clahe = cv2.createCLAHE(**kwargs)
    lab[0] = clahe.apply(lab[0])
    lab = cv2.merge((lab[0], lab[1], lab[2]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gabor(img, **kwargs):
    g_kernel = cv2.getGaborKernel(**kwargs)
    return cv2.filter2D(img, cv2.CV_8UC3, g_kernel.sum())


def increase_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v += 255
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 0.3, theta, 9.0, 0.6, 50, ktype=cv2.CV_32F)
    kern /= 1.5*kern.sum()
    filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum

def main():
    with video_capture(VIDEO) as cap:
        net = cv2.dnn.readNet(WEIGHTS, CONFIG)
        frame_id = -1
        belt_detected = BeltDetected()
        while True:
            frame = cap.read()
            frame_id += 1
            if not frame[0]:
                break
            img = frame[1][:, 50: -50]

            # TODO: your code here
            img = increase_brightness(img)

            img = apply_clahe(img=img, clipLimit=5, tileGridSize=(17, 17))
            # better results for corner belt, slightly worse results for main part
            # img = apply_gabor(img=img, ksize=(4, 4), sigma=5, theta=89,
            #                   lambd=1, gamma=2, psi=0, ktype=cv2.CV_64F)
            # the best result for main part
            img = apply_gabor(img=img, ksize=(31, 31), sigma=2.9, theta=160,
                              lambd=14.5, gamma=35, psi=50, ktype=cv2.CV_64F)
            belt_detected = belt_detector(net, img, belt_detected, frame_id)
            cv2.imshow("Image", img)

            # to decide which frame should be assumed as belt position changing
            # chosen id=124, although it is arguable
            # if frame_id in range(120, 126):
            #     cv2.imwrite(SAVE_PATH + "{id}.png".format(id=frame_id), img)

            key = cv2.waitKey(1)
            if key == 27:
                break
        print_belt_report(belt_detected, frame_id)

=======
    
    net =cv2.dnn.readNet("YOLOFI2.weights","YOLOFI.cfg")
    cap = cv2.VideoCapture("test.mp4")
    classes=[]  
    l=1
    with open("obj.names","r")as f:
            classes = [line.strip()for line in f.readlines()]
            layers_names = net.getLayerNames()
            outputlayers= [layers_names[i[0]-1]for i in net.getUnconnectedOutLayers()]
            colors = np.random.uniform(0,255,size =(len(classes),3))
            font = cv2.FONT_HERSHEY_PLAIN
            frame_id=0  
            dd =-1
            time_now=time.time()
            frame_id=0
            err=0

            count = 0 # for counting frames

            while True:
                _, frame = cap.read()
                frame_id += 1           
                beltcornerdetected = False
                beltdetected = False 
                height , width , channels = frame.shape


                #Type you code here

                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))

                R, G, B = cv2.split(frame)


                output1_R = clahe.apply(R)
                output1_G = clahe.apply(G)
                output1_B = clahe.apply(B)

                frame = cv2.merge((output1_R, output1_G, output1_B))

                cv2.fastNlMeansDenoising(frame,frame,3,5,11)

                filters = build_filters()
                frame = process(frame, filters)

                blob = cv2.dnn.blobFromImage(frame, 0.00392, (480,480),(0,0,0),True,crop= False)
                net.setInput(blob)
                outs = net.forward(outputlayers)
                class_ids=[]
                boxes=[]
                shape=[]
                confidence=0

                for out in outs:

                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        if confidence> 0.2:
                            center_x= int(detection[0] *width)
                            center_y=int(detection[1]* height)
                            w= int(detection[2] *width)
                            h= int(detection[3] * height)
                            x= int(center_x- w /2)
                            y= int(center_y -h /2)
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                            if class_id== 1:
                                beltcornerdetected=True
                            elif class_id == 0:
                                beltdetected=True
                            
                print(count, ' ', beltdetected)
                count+=1
                cv2.imshow("Image",frame)
                key =cv2.waitKey(1)
                if key == 27:
                  break
           
            cap.release()    
            cv2.destroyAllWindows()
       
if __name__ == '__main__':
        main()


if __name__ == "__main__":
    main()
