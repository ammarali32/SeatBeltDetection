import numpy as np 
import cv2
import os
import time
import numpy as np
from imutils import face_utils
import imutils


# def conservative_smoothing_gray(data, filter_size):
# МЕДЛЕННО РАБОТАЕТ И НЕ ОЧЕНЬ ЭФФЕКТИВНО (ВРОДЕ)!
#     temp = []
#     indexer = filter_size // 2
#     new_image = data.copy()
#     nrow, ncol, nchan = data.shape
#
#     for i in range(nrow):
#         for j in range(ncol):
#             for k in range(i - indexer, i + indexer + 1):
#                 for m in range(j - indexer, j + indexer + 1):
#                     if (k > -1) and (k < nrow):
#                         if (m > -1) and (m < ncol):
#                             temp.append(data[k, m].all())
#             temp.remove(data[i, j].all())
#
#             max_value = max(temp)
#             min_value = min(temp)
#
#             if data[i, j].all() > max_value:
#                 new_image[i, j] = max_value
#             elif data[i, j].all() < min_value:
#                 new_image[i, j] = min_value
#             temp = []
#     return new_image.copy()

def filterClahe(img):
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    R, G, B = cv2.split(img)

    clahe_R = clahe.apply(R)
    clahe_G = clahe.apply(G)
    clahe_B = clahe.apply(B)

    img = cv2.merge((clahe_R, clahe_G, clahe_B))

    cv2.fastNlMeansDenoising(img)
    return img

def changingImage(img):

    # brightness = 30
    # contrast = 30
    # img = np.int16(img)
    # # img = img * (contrast / 127 + 1) - contrast + brightness
    # img = np.clip(img, 0, 255)
    # img = np.uint8(img)

    # dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    # bg_img = cv2.medianBlur(dilated_img, 21)
    # diff_img = 255 - cv2.absdiff(img, bg_img)
    # img = diff_img.copy()

    #kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    # img = cv2.filter2D(img, -1, kernel)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = filterClahe(img)

    return img

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
                if frame is None:
                    break
                height , width , channels = frame.shape

                #Type you code here
                # frame = conservative_smoothing_gray(frame, 5)
                frame = changingImage(frame)

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

