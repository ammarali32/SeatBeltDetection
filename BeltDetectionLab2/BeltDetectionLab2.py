import numpy as np 
import cv2
import os
import time
import numpy as np
from imutils import face_utils
import imutils

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

                #Type you code here
                clahe = cv2.createCLAHE(clipLimit=299.0, tileGridSize=(8,8))
                output = cv2.split(frame)
                result = []
                for element in output:
                    result.append(clahe.apply(element))
                frame =  cv2.merge(result)
                filters = []
                ksize = 31
                for theta in np.arange(0, np.pi, np.pi / 16):
                    kern = cv2.getGaborKernel((ksize, ksize), 2.07, theta, 22, 66, 5, ktype=cv2.CV_32F)
                kern /= 1.5*kern.sum()
                filters.append(kern)
                accum = np.zeros_like(frame)
                for kern in filters:
                    fimg = cv2.filter2D(frame, cv2.CV_8UC3, kern)
                    np.maximum(accum, fimg, accum)
                frame = accum


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
                            
                print(frame_id, beltdetected)
                cv2.imshow("Image",frame)
                key =cv2.waitKey(1)
                if key == 27:
                  break
           
            cap.release()    
            cv2.destroyAllWindows()
       
if __name__ == '__main__':
        main()

