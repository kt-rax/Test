# -*- coding: utf-8 -*-

import torch
import torchvision
from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import  random
import os
from torchvision import models
from timeit import Timer
import logging
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from keras.utils import plot_model
from PennFudanPed import PennFudanDataset 

device = torch.device("cpu") 
#model = torch.load(r'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',map_location = torch.device('cpu'))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device) 


'''
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False).to(device)
# pretrained = False 
images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []

for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)

#output = model(images, targets)
pred2 = torch.load('pred.pt', map_location=torch.device('cpu'))
#pred3 = torch.tensor(pred2)
make_dot(pred2[0], params=dict(model.named_parameters()),show_attrs=True, show_saved=True).render("Model_plot", format="png")
'''

# model visual test
model2 = model
#plot_model(model,to_file='chapter10.3.png',show_shapes=True) 
#print(model2)

#model = torch.jit.load('fasterrcnn_resnet50_fpn_coco-258fb6c6.pth').eval().to(device)
#model = torch.hub.load(r'/','fasterrcnn_resnet50_fpn_coco-258fb6c6',pretrained=True)
#model = torch.load('fasterrcnn_resnet50_fpn_coco-258fb6c6.pth')
#model.eval()
#pip install torch==1.9.0+cpu torchvision==0.4.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
#pip torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html 
#pip install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html  
#for param in model.parameters():#    
#    param.required_grad = False

#??????????????????????????????????????? torch==1.2.0+cpu torchvision==0.4.0+cpu?????????????????????????????????????????????????????????????????????cpu???????????? ????????????
#pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html 

#device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# ???????????? 
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

# ?????????????????????
model.eval()


# ??????Pythorch????????????????????????????????????'N/A'??????????????????????????? 
COCO_INSTANCE_CATEGORY_NAMES = ['__background__','person','bicycle','car','motorcycl','airplane','bus','train','truck',
                                'boat','traffic light','fire hydrant','N/A','stop sign','parking meter','bench','biard',
                                'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','N/A','backpack',
                                'umbrella','N/A','N/a','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
                                'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
                                'N/A','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
                                'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
                                'bed','N/A','dining talbe','N/A','N/A','toilet','N/A','tv','laptop','mouse','remote',
                                'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','N/A','book',
                                'clock','vase','scissors','teddy bear','hair drier','toothbrush']

 
    
    # ?????? Pytorch ???????????????????????????????????? 'N/A' ????????????????????????
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

#  ?????????????????????????????????
def get_prediction(img_path, threshold):
    img = Image.open(img_path)  # Load the image  ????????????
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img)  # Apply the transform to the image  ????????? torch ??????
    pred = model([img.to(device)])  # Pass the image to the model  ????????????
    # ?????????????????? 
    make_dot(pred[0]['boxes'], params=dict(model.named_parameters()),show_attrs=True, show_saved=True).render("fater_resnet50_fpn", format="png")
    
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score  ?????????????????????
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes  ???????????????????????????
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())  #  ???????????????????????????

    # Get list of index with score greater than threshold.
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  #  ??????????????????????????????????????????????????????
    #  ?????????????????????????????????????????????????????????????????????????????????????????????????????????
    pred_boxes = pred_boxes[:pred_t+1]  
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


#  ??????????????????????????????????????????
def object_detection_api(img_path, threshold=0.5, rect_th=1, text_size=0.5, text_th=1):
    boxes, pred_cls = get_prediction(img_path, threshold)  # Get predictions
    
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    result_dict = {}  #  ??????????????????????????????????????????
    for i in range(len(boxes)):
        color = tuple(random.randint(0, 255) for i in range(3))
        cv2.rectangle(img,
                      (int(boxes[i][0][0]),int(boxes[i][0][1])),  #boxes[i][0],fixed  error: (-5:Bad argument) - Can't parse 'pt1'. Sequence item with index 0 has a wrong type
                      (int(boxes[i][1][0]),int(boxes[i][1][1])),  #boxes[i][1],   
                      color=color,
                      thickness=rect_th)  # Draw Rectangle with the coordinates

        cv2.putText(img,
                    pred_cls[i],
                    (int(boxes[i][0][0]),int(boxes[i][0][1])),    #boxes[i][0], fixedd - Can't parse 'org'. Sequence item with index 0 has a wrong type
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    text_size,  
                    color,
                    thickness=text_th)  # Write the prediction class

        #  ????????????????????????????????????????????????
        if pred_cls[i] not in result_dict:
            result_dict[pred_cls[i]] = 1
        else:
            result_dict[pred_cls[i]] += 1
        print(result_dict)
    plt.figure(figsize=(6.4, 4.8))  # display the output image
    #(float, float), default: rcParams["figure.figsize"] (default: [6.4, 4.8]) Width, height in inches.
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('result.jpg')
    plt.show()
    

if __name__ == "__main__":
    object_detection_api('./eagle.jpg', threshold=0.5)
    FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'   
'''
    #??????5?????????????????????????????? 
    logging.basicConfig(format=FORMAT)
    d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
    for i in range(1,5):
        logger = logging.getLogger('??????')
        logger.warning('' ,extra=d)
        object_detection_api('./people.jpg', threshold=0.5)
    #Timer(object_detection_api('./Test_E1.jpg', threshold=0.5)).timeit()
'''
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        