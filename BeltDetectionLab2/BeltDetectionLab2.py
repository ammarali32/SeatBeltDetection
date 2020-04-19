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

            count = 0 # for counting frames

            while True:
                _, frame = cap.read()
                frame_id += 1           
                beltcornerdetected = False
                beltdetected = False 
                height , width , channels = frame.shape
                
                frame = cv2.fastNlMeansDenoisingColored(frame, None, 1, 7, 21)  #After that, the program stopped selecting on the jacket pocket, but select the car door, that's bad
                #If the denoising is too strong, the belt cannot be detected. This is not the best solution.

                #frame = cv2.medianBlur(frame, 5)
                #frame = cv2.bilateralFilter(frame, 9, 75, 75)  --  i tried this. It's not working. I think imagage filtering is not very suitable.

                g_kernel = cv2.getGaborKernel((25, 25), 0.1, 1*np.pi/2, 9.0, 0.6, 25, ktype=cv2.CV_32F) #After the Gabor filter, it got better (there is almost no "false" in the output), but there is a false select on the door.
                frame = cv2.filter2D(frame, cv2.CV_8UC3, g_kernel)

                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab_planes = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                lab_planes[0] = clahe.apply(lab_planes[0])
                lab = cv2.merge(lab_planes)
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) #This is the best solution. Door problem is resolved. Thank you, CLAHE!



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

