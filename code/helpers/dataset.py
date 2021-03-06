from glob import glob

import matplotlib.patches as patches
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# For reference on how to e.g. visualize the masks see: https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/davis.py
# Reference how to compute the bounding boxes: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


class DAVISDataset(Dataset):
    def __init__(self, root, subset='train', resolution='480p', transforms=None, year='2017', sequences='all'):
        self.root = root
        self.subset = subset
        self.img_path = os.path.join(self.root, 'JPEGImages', resolution)
        self.mask_path = os.path.join(self.root, 'Annotations', resolution)
        self.imagesets_path = os.path.join(self.root, 'ImageSets', year) if year == '2017' else os.path.join(self.root,
                                                                                                             'ImageSets',
                                                                                                             resolution)
        self.transforms = transforms

        if sequences == 'all':
            with open(os.path.join(self.imagesets_path, f'{self.subset}.txt'), 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp] if year == '2017' else sorted(
                {x.split()[0].split('/')[-2] for x in tmp})
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = []
        for seq in sequences_names:
            info = {}
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            info['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            info['masks'] = masks
            info['name'] = seq
            self.sequences.append(info)

        # self.random_horizontal_flip = data_aug.RandomHorizontalFlip()
        # self.scale = data_aug.RandomScale(scale=0.05)
        # self.rotate = data_aug.RandomRotate(angle=5)

    def __len__(self):
        return len(self.sequences)

    def apply_augmentations(self, imgs, masks, gt_boxes):
        raise NotImplementedError
        self.rotate.reset()
        self.scale.reset()
        self.random_horizontal_flip.reset()

        for idx in range(len((imgs))):
            img, img_masks, img_gt_boxes = imgs[idx], masks[idx], gt_boxes[idx]
            img_masks = [np.expand_dims(mask, axis=-1) for mask in img_masks]
            img, img_masks, img_gt_boxes = self.random_horizontal_flip(img, img_masks, np.array(img_gt_boxes).astype(np.float64))
            img, img_masks, img_gt_boxes = self.scale(img, img_masks, img_gt_boxes)
            img, img_masks, img_gt_boxes = self.rotate(img, img_masks, img_gt_boxes)
            if len(img_gt_boxes) > 0:
                img_boxes = [list(img_gt_boxes[i, :].astype(np.int64)) for i in range(len(img_gt_boxes))]
            else:
                img_boxes = []

            img_masks = [mask[:, :, 0].astype(np.bool) for mask in img_masks]

            imgs[idx] = img
            masks[idx] = img_masks
            gt_boxes[idx] = img_boxes

        return imgs, masks, gt_boxes

    # returns a whole sequence
    def __getitem__(self, idx):
        imgs = []
        masks = []
        boxes = []
        seq_name = self.sequences[idx]['name']

        for img, msk in zip(self.sequences[idx]['images'], self.sequences[idx]['masks']):
            image = Image.open(img)
            image = np.array(image)
            mask = Image.open(msk)
            mask = np.array(mask)
            imgs.append(image)

            # get ids of the different objects in the img
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            # split the color-encoded mask into a set of binary masks
            binary_masks = mask == obj_ids[:, None, None]
            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            img_masks = []
            img_boxes = []
            for i in range(num_objs):
                pos = np.where(binary_masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if xmin < xmax and ymin < ymax:
                    img_boxes.append([xmin, ymin, xmax, ymax])
                    img_masks.append(binary_masks[i])

            masks.append(img_masks)
            boxes.append(img_boxes)
        #
        # if self.subset == 'train':
        #     imgs, masks, boxes = self.apply_augmentations(imgs, masks, boxes)

        targets = []
        for i in range(len(imgs)):
            target = {}
            if len(boxes[i]) == 0:
                targets.append(target)
                continue
            bxs = torch.as_tensor(boxes[i], dtype=torch.float32)
            target["boxes"] = bxs
            target["labels"] = torch.ones((len(bxs),), dtype=torch.int64)
            target["masks"] = torch.as_tensor(masks[i], dtype=torch.uint8)
            target["image_id"] = torch.tensor([1000 * idx + i])  # unique if no seq is longer than 1000 frames
            target["area"] = (bxs[:, 3] - bxs[:, 1]) * (bxs[:, 2] - bxs[:, 0])
            target["iscrowd"] = torch.zeros((len(bxs),), dtype=torch.int64)

            targets.append(target)

        targets = tuple(targets)

        if self.transforms:
            for img_idx in range(len(imgs)):
                imgs[img_idx] = self.transforms(imgs[img_idx].copy())

        # visualize_image_with_properties(imgs[0].cpu().numpy().transpose(1, 2, 0), masks=targets[0]['masks'], boxes=targets[0]['boxes'],)

        return imgs, targets, seq_name


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    subsets = ['train', 'val']
    ds = DAVISDataset(root='data/DAVIS', subset='val')
    dataloader = DataLoader(ds, batch_size=1)

    colors = ['r', 'b']
    for seq in dataloader:
        for image, masks, boxes in zip(seq[0], seq[1], seq[2]):
            ax = plt.subplot(1, 1, 1)
            ax.imshow(image.squeeze())
            i = 0
            for mask, box in zip(masks, boxes):
                # ax = plt.subplot(plot_count, 1, 2+i)
                ax.imshow(mask.squeeze(), alpha=0.4)
                i += 1

                x = box[0][0].item()
                y = box[0][1].item()
                width = box[0][2].item() - x
                height = box[0][3].item() - y
                rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

            plt.show(block=True)
