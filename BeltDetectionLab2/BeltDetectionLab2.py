import cv2
import os
import time
import numpy as np
from imutils import face_utils
import imutils


def apply_clahe(img):
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16))
    clahe_img = clahe.apply(img)
    # clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return clahe_img


def equalize_histogram(img):
    equ = cv2.equalizeHist(img)
    return equ


def apply_gabor(img):
    g_kernel = cv2.getGaborKernel((25, 25), 3, 162, 15, 35, 50, cv2.CV_64F)
    return cv2.filter2D(img, cv2.CV_8UC3, g_kernel.sum())


def apply_blur(img):
    # blur_img = cv2.blur(img, (3, 3))
    # blur_img = cv2.medianBlur(img, 3)
    # blur_img = cv2.bilateralFilter(img, 10, 25, 25)
    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    return blur_img


def main():

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
            while True:
                _, frame = cap.read()
                frame_id += 1
                beltcornerdetected = False
                beltdetected = False
                height , width , channels = frame.shape

                frame = apply_blur(frame)

                r_part, g_part, b_part = cv2.split(frame)

                r_output = apply_clahe(r_part)
                # r_output = equalize_histogram(r_part)

                g_output = apply_clahe(g_part)
                # g_output = equalize_histogram(g_part)

                b_output = apply_clahe(b_part)
                # b_output = equalize_histogram(b_part)

                frame = cv2.merge((r_output, g_output, b_output))

                frame = apply_gabor(frame)
                cv2.fastNlMeansDenoising(frame, frame, 3, 7, 21)

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

                print(beltdetected)
                cv2.imshow("Image",frame)
                key =cv2.waitKey(1)
                if key == 27:
                  break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
        main()
