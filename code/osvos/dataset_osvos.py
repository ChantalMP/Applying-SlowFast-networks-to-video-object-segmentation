from glob import glob
from math import ceil

import numpy as np
import os
import torch
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from DataAugmentationForObjectDetection.data_aug import data_aug


class DAVISSequenceDataset(Dataset):
    def __init__(self, root, sequence_name, fast_pathway_size, transforms=None, resolution='480p', cfg=None):
        self.root = root
        self.sequence_name = sequence_name
        self.fast_pathway_size = fast_pathway_size
        self.img_path = os.path.join(self.root, 'JPEGImages', resolution)
        self.mask_path = os.path.join(self.root, 'Annotations', resolution)
        self.imagesets_path = os.path.join(self.root, 'ImageSets', resolution)

        self.transforms = transforms

        self.sequence_info = {}
        images = np.sort(glob(os.path.join(self.img_path, sequence_name, '*.jpg'))).tolist()
        self.sequence_info['images'] = images
        masks = np.sort(glob(os.path.join(self.mask_path, sequence_name, '*.png'))).tolist()
        self.sequence_info['masks'] = masks
        self.sequence_info['name'] = sequence_name

        self.random_horizontal_flip = data_aug.RandomHorizontalFlip()
        if cfg is None:
            self.scale = data_aug.RandomScale(scale=(0.25))
        else:
            self.scale = data_aug.RandomScale(scale=(cfg.scale))
        self.rotate = data_aug.RandomRotate(angle=30)

    def __len__(self):
        return 200

    def apply_augmentations(self, img, img_masks=None, img_gt_boxes=None):
        '''
        :param img: imgs if only for neigbours else one image only and masks and boxes exist
        '''

        if img_masks is None or img_gt_boxes is None:
            for idx in range(len(img)):
                current_img = img[idx]
                current_img, _, _ = self.random_horizontal_flip(current_img)
                current_img, _, _ = self.scale(current_img)
                current_img, _, _ = self.rotate(current_img)
                img[idx] = current_img
            return img, None, None

        img_masks = [np.expand_dims(mask, axis=-1) for mask in img_masks]
        img, img_masks, img_gt_boxes = self.random_horizontal_flip(img, img_masks, np.array(img_gt_boxes).astype(np.float64))
        img_tmp, img_masks_tmp, img_gt_boxes_tmp = self.scale(deepcopy(img), deepcopy(img_masks), deepcopy(img_gt_boxes))
        while len(img_gt_boxes_tmp) == 0:
            self.scale.reset()
            img_tmp, img_masks_tmp, img_gt_boxes_tmp = self.scale(deepcopy(img), deepcopy(img_masks), deepcopy(img_gt_boxes))
        img, img_masks, img_gt_boxes = img_tmp, img_masks_tmp, img_gt_boxes_tmp
        img, img_masks, img_gt_boxes = self.rotate(img, img_masks, img_gt_boxes)

        img_masks = [mask[:, :, 0].astype(np.bool) for mask in img_masks]

        return img, img_masks, img_gt_boxes

    # returns the first image of the sequence
    def __getitem__(self, idx):

        self.rotate.reset()
        self.scale.reset()
        self.random_horizontal_flip.reset()

        image_paths = self.sequence_info['images'][0:ceil(self.fast_pathway_size / 2)]  # get first frame and neigbors
        mask_path = self.sequence_info['masks'][0]

        imgs = []

        for img in image_paths:
            image = Image.open(img)
            image = np.array(image)
            imgs.append(image)

        middle_image = imgs[0]
        mask = Image.open(mask_path)
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
        assert len(img_boxes) > 0
        middle_image, img_masks, img_boxes = self.apply_augmentations(middle_image, img_masks, img_boxes)

        target = {}
        bxs = torch.as_tensor(img_boxes, dtype=torch.float32)
        target["boxes"] = bxs
        target["labels"] = torch.ones((len(bxs),), dtype=torch.int64)
        target["masks"] = torch.as_tensor(img_masks, dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx])  # unique if no seq is longer than 1000 frames
        target["area"] = (bxs[:, 3] - bxs[:, 1]) * (bxs[:, 2] - bxs[:, 0])
        target["iscrowd"] = torch.zeros((len(bxs),), dtype=torch.int64)

        # augment neighboring frames
        imgs, _, _ = self.apply_augmentations(imgs)
        if self.transforms:
            for img_idx in range(len(imgs)):
                imgs[img_idx] = self.transforms(imgs[img_idx].copy())

        # padding in front
        padding_count = self.fast_pathway_size // 2
        if padding_count > 0:
            reversed_frames = list(reversed(imgs[1:]))  # TODO only tested for odd numbers
            assert padding_count == len(reversed_frames)
            # imgs = torch.cat([torch.zeros_like(imgs[0].repeat(padding_count, 1, 1, 1)), torch.stack(imgs)])
            imgs = torch.cat([torch.stack(reversed_frames), torch.stack(imgs)])
            imgs = [elem for elem in imgs]

        return imgs, target


if __name__ == '__main__':

    ds = DAVISSequenceDataset(root='data/DAVIS_2016', sequence_name='camel', fast_pathway_size=7, transforms=Compose([ToTensor()]))

    img, target = ds[0]
