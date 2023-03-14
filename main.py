from dataset import PenDudanPedDataset
import numpy as np
import torch
import sys

if __name__ == '__main__':
    # dataset = PenDudanPedDataset(root='Data\PennFudanPed', transforms=None)
    # mask = dataset.__getitem__(0)
    # # for i in range(len(mask)):
    # #     for j in range(len(mask[i])):
    # #         print(mask[i][j], " ", end="")
    # #     print()
    
    # mask = np.array([[0, 3, 2], [0, 1, 0], [0, 1, 1]])
    # # instances (objects) are encoded as different colors
    # obj_ids = np.unique(mask)
    
    # # first is background, so remove it
    # obj_ids = obj_ids[1:]
    
    # # split the color-encoded mask into a set of binary mask
    # masks = mask == obj_ids[:, None, None]
    # masks = torch.as_tensor(masks, dtype = torch.uint8)
    # sys.path.remove("/vision/references/detection")
    print(sys.path)
    
