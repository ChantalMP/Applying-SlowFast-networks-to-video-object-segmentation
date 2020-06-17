import os
from glob import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

import torch
from DataAugmentationForObjectDetection.data_aug import bbox_util, data_aug
from copy import deepcopy


class DAVISSequenceDataset(Dataset):
    def __init__(self, root, sequence_name, resolution='480p', year='2016'):
        self.root = root
        self.sequence_name = sequence_name
        self.img_path = os.path.join(self.root, 'JPEGImages', resolution)
        self.mask_path = os.path.join(self.root, 'Annotations', resolution)
        self.imagesets_path = os.path.join(self.root, 'ImageSets', year) if year == '2017' else os.path.join(self.root,
                                                                                                             'ImageSets',
                                                                                                             resolution)
        loading_str = f'predicted_proposals_val_{year}.pt'  # only use validation data for osvos anyway
        self.box_proposals = torch.load(f'maskrcnn/{loading_str}')[self.sequence_name]

        self.sequence_info = {}
        images = np.sort(glob(os.path.join(self.img_path, sequence_name, '*.jpg'))).tolist()
        self.sequence_info['images'] = images
        masks = np.sort(glob(os.path.join(self.mask_path, sequence_name, '*.png'))).tolist()
        self.sequence_info['masks'] = masks
        self.sequence_info['name'] = sequence_name

        self.random_horizontal_flip = data_aug.RandomHorizontalFlip()
        self.scale = data_aug.RandomScale(scale=(.75, 1.25))
        self.rotate = data_aug.RandomRotate(angle=30)

    def __len__(self):
        return 1

    def expand_proposals(self, proposals, img_width, img_height, ratio=0.1):
        for i in range(len(proposals)):
            box = proposals[i]
            width_change = (box[2] - box[0]) * ratio
            height_change = (box[3] - box[1]) * ratio
            bigger_box = torch.tensor(
                [max(0, box[0] - width_change), max(0, box[1] - height_change), min(img_width, box[2] + width_change),
                 min(img_height, box[3] + height_change)])
            proposals[i] = bigger_box

        return proposals

    # returns the first image of the sequence
    def __getitem__(self, idx):
        image = self.sequence_info['images'][0]  # or which frame we want to use for finetuning
        image = Image.open(image)
        image = np.array(image)
        mask = self.sequence_info['masks'][0]
        mask = Image.open(mask)
        mask = np.array(mask)

        # get ids of the different objects in the img
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set of binary masks
        binary_masks = mask == obj_ids[:, None, None]

        img_masks = []
        img_boxes = []
        pos = np.where(binary_masks[0])  # we use only one object
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if xmin < xmax and ymin < ymax:
            img_boxes.append([xmin, ymin, xmax, ymax])
            img_masks.append(binary_masks[0])

        # transform img, mask and box
        img, mask, box = self.random_horizontal_flip(image, np.expand_dims(img_masks[0], axis=2),
                                                     np.array(img_boxes).astype(np.float64))

        assert len(box) > 0
        scaled_img, scaled_mask, scaled_box = self.scale(deepcopy(img), deepcopy(mask.astype(np.uint8)), deepcopy(box))
        while len(scaled_box) == 0:
            scaled_img, scaled_mask, scaled_box = self.scale(deepcopy(img), deepcopy(mask.astype(np.uint8)),
                                                             deepcopy(box))
        img, mask, box = scaled_img, scaled_mask, scaled_box

        img, mask, box = self.rotate(img, mask, box)
        img_masks = [mask[:, :, 0]]
        img_boxes = [list(box[0, :].astype(np.int64))]


        target = {}
        bxs = torch.as_tensor(img_boxes, dtype=torch.float32)
        target["boxes"] = bxs
        target["labels"] = torch.ones((len(bxs),), dtype=torch.int64)
        target["masks"] = torch.as_tensor(img_masks, dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx])  # unique if no seq is longer than 1000 frames
        target["area"] = (bxs[:, 3] - bxs[:, 1]) * (bxs[:, 2] - bxs[:, 0])
        target["iscrowd"] = torch.zeros((len(bxs),), dtype=torch.int64)
        target["proposals"] = self.box_proposals.cpu()

        # TODO apply transforms to image, mask and box and proposals
        # TODO get previous and following frames
        # TODO visualize augmentations
        # TODO adapt methods to also augment proposals and neighbouring frames

        return image, target


if __name__ == '__main__':
    from torchvision import transforms
    from osvos import osvos_transforms as tr

    # Transforms from OSVOS paper:
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    ds = DAVISSequenceDataset(root='data/DAVIS_2016', sequence_name='camel')

    img, target = ds[0]
