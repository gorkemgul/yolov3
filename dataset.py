import config
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (iou_width_height as iou, non_max_supression as nms)

# To don't get any errors while loading images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, img_size, S = [13, 26, 52], classes = 20, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0], anchors[1], anchors[2]) # to put all anchors together
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.classes = classes
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname = label_path, delimiter = " ", ndmin = 2), 4, axis = 1).tolist()
        img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        img = np.array(Image.open(img_path).convert("RGB")) # for albumentations turn in into an array

        if self.transform:
            augmentations = self.transform(image = img, bboxes = bboxes) # bboxes will also be transformed according to operations
            img = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for bbox in bboxes:
            iou_anchors = iou(torch.tensor(bbox[2:4]), self.anchors)
            anchors_indices = iou_anchors.argsort(descending = True, dim = 0)
            x, y, width, height, class_label = bbox
            has_anchor = [False, False, False]

            for anchor_idx in anchors_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coords = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coords
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
