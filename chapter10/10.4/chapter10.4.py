# -*- coding: utf-8 -*-
import os
import time
import cv2
import numpy as np
import YOLO

def process_image(img):
    """Resize,reduce and expand image.
    # Arguument:
        img: original imgae.
    # Returns
        imgae: ndarray(64,64,3),processed image
    """
    
    image = cv2.resize(img,(416,416),interpolation=cv2.INTER_CUBIC)
    image = np.array(image,dtype='float32')
    image /= 255.
    image = np.expand_dims(image,axis=0)
    return image

def get_classes(file):
    """ Get classes name.
    # Argument:
        file: classes name for database
    # Returns:
        calss_name: List,classes name
    """
    with open(file) as f:
        class_names = f.readlines()
    
    class_names = [c.strip() for c in class_names]
    return class_names

def draw(image,boxes,scores,classes,all_classes):
    """ Draw the boxes on the image
    # Argument:
        image: original image.
        boxes: ndarray,boxes of object.
        classes: ndarray,classes of object.
        scores: ndarray,scores of object.
        all classes: all classes name.
    """
    for box,score,cl in zip(boxes,scores,classes):
        x,y,w,h = box
        top = max(0,np.floor(x+0.5).astype(int))
        left = max(0,np.floor(y+0.5).astype(int))
        right = min(image.shape[1],np.floor(x+w+0.5).astype(int))
        bottom = min(image.shape[0],np.floor(y+h+0.5).astype(int))
        cv2.rectangle(image,(top,left),(right,bottom),(255,0,0),2)
        cv2.putText(image,'{0}{1:.2f}'.format(all_classes[cl],score),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.IINE_AA)
        
        print('class:{0},socre:{1:.2f}'.format(all_classes[cl],score))
        print('box coordinate x,y,w,h:{0}'.format(box))
    print()

def detect_image(image,yolo,all_classes):
    """Use yolo v3 to detect images.
    # Argument:
        image: original image
        yolo: YOLO,yolo model
        all_classes: all classes name.
    # Returns:
        image: processed image.
    """
    pimage = process_image(image)
    start = time.time()
    boxes,classes,scores = yolo.predict(pimage,image.shape)
    end = time.time()
    print('time:{0:2f}s'.format(end-start))
    if boxes is not None:
        draw(image,boxes,scores,classes,all_classes)
    return image

def detect_video(video,yolo,all_classes):
    """use yolo v3 to detcet video.
    #Argument:
        video: video file.
        yolo: YOLO,yolo model
        all_classes: all classes name.
    """
    video_path = os.path.join('videos','test',video)
    camera = cv2.VideoCapture(video_path)
    cv2.namedWindow('detection',cv2.WINDOW_AUTOSIZE)
    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FFAME_WIDTH)),int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    vout = cv2.VideoWriter()
    vout.open(os.path.join('videos','res',video),fourcc,20,sz,True)
    
    while True:
        res,frame = camera.read()
        if not res:
            break
        image = detect_image(frame,yolo,all_classes)
        cv2.imshow('detection',image)
        #save the video frame by frame
        vout.write(image)
        if cv2.waitKey(110) & 0xff == 27:
            break
        
    vout.release()
    camera.release()
    
if __name__ == '__main__':
    yolo = YOLO.YOLO(0.6,0.5)
    file = 'keras-yolo3-master/keras-yolo3-master/model_data/coco_classes.txt'
    all_classes = get_classes(file)
    # detect images in test folder. 
    for (root,dirs,files) in os.walk('images/test'):
        if files:
            for f in files:
                print(f)
                path = os.path.join(root,f)
                image = cv2.imread(path)
                image = detect_image(image,yolo,all_classes)
                cv2.imwrite('images/result/'+f,image)
    # detect videos one at a time in videos/test folder
    # video = 'library1.mp4'

#detect_video(video,yolo,all_classes)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    