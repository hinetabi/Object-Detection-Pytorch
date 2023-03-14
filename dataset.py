import torch
import os
import numpy as np
from PIL import Image


class PenDudanPedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        
        # load all images files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root,"PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root,"PedMasks"))))

    def __getitem__(self, idx):
        """
        RETURN 
        - boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
        - labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
        - image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
        - area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
        - iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
        - (optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
        - (optionally) keypoints (FloatTensor[N, K, 3]): For each one of the N objects, it contains the K keypoints in 
        [x, y, visibility] format, defining the object. visibility=0 means that the keypoint is not visible. 
        
        Note that for data augmentation, the notion of flipping a keypoint is dependent on the data representation, 
        and you should probably adapt references/detection/transforms.py for your new keypoint representation
        """
        
        # load images and marks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        
        # not convert mask to RGB because each color corresponds to a different instance
        # with 0 is the background
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # instances (objects) are encoded as different colors
        obj_ids = np.unique(mask)
        
        # first is background, so remove it
        obj_ids = obj_ids[1:]
        
        
        # split the color-encoded mask into a set of binary mask
        masks = mask == obj_ids[:, None, None]
        
        num_objs = len(obj_ids)
        # get bounding box for each instances
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        # convert every thing into torch.Tensor
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        
        # there is only one class
        labels = torch.ones((num_objs,),dtype = torch.int64)
        masks = torch.as_tensor(masks, dtype = torch.float32)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,),dtype = torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
        
        
    def __len__(self):
        return len(self.imgs)