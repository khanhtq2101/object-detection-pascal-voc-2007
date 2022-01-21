import torch

class_to_idx = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
                'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,
                'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19
}
idx_to_class = {i:c for c, i in class_to_idx.items()}

def xml_to_tensor(annotation):
  '''
  annotation: annotation of image
  '''
  
  bbox =[]
  for object_info in annotation['annotation']['object']:
    object_bbox = object_info['bndbox']
    for name in object_bbox.keys():
      object_bbox[name] = int(object_bbox[name])
    object_bbox = [object_bbox['xmin'], object_bbox['xmax'], object_bbox['ymin'], object_bbox['ymax']]
    class_idx = [class_to_idx[object_info['name']]]

    bbox.append(object_bbox + class_idx)
  
  return torch.tensor(bbox)
