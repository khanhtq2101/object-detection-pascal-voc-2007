import cv2
import numpy as np

class_to_idx = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
                'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,
                'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19
}
idx_to_class = {i:c for c, i in class_to_idx.items()}

def hello_vis():
  print('hien')
  print('how can i meet you')

def detection_visulizer(img, bbox= None, pred_bbox = None):
  '''
  img: PIL image
  bbox: (x_min, x_max, y_min, y_max, class)
        shape Nx5, where N is number of objects in image
  '''

  copy_img = np.array(img).astype('uint8')
  
  if bbox is not None:
    for one_bbox in bbox:
      start_point = (one_bbox[0], one_bbox[2])
      end_point = (one_bbox[1], one_bbox[3])
      
      copy_img = cv2.rectangle(copy_img, start_point, end_point, (0, 255, 0), 2)

      object_name = idx_to_class[one_bbox[4].item()]

      text_size = cv2.getTextSize(object_name, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, thickness = 1)[0]
      copy_img = cv2.rectangle(copy_img, start_point, (start_point[0] + text_size[0], start_point[1] + text_size[1]), (0, 255, 0), -1)

      copy_img = cv2.putText(copy_img, object_name, (start_point[0], start_point[1] + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4, color = (0, 0, 0), thickness = 1)

  if pred_bbox is not None:
    for one_bbox in pred_bbox:
      start_point = (one_bbox[0], one_bbox[2])
      end_point = (one_bbox[1], one_bbox[3])
      
      copy_img = cv2.rectangle(copy_img, start_point, end_point, (0, 255, 0), 2)
      
  return copy_img