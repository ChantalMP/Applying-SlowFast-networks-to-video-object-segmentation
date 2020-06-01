import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import matplotlib.patches as patches
import torch
from math import ceil
from tqdm import tqdm


# For reference on how to e.g. visualize the masks see: https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/davis.py
# Reference how to compute the bounding boxes: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

class DAVISDataset(Dataset):
    def __init__(self, root, subset='train', resolution='480p', transforms=None, max_seq_length=40, fast_pathway_size=16):
        self.root = root
        self.subset = subset
        self.img_path = os.path.join(self.root, 'JPEGImages', resolution)
        self.mask_path = os.path.join(self.root, 'Annotations', resolution)
        self.imagesets_path = os.path.join(self.root, 'ImageSets', '2017')
        self.transforms = transforms
        self.max_seq_length = max_seq_length
        self.fast_pathway_size = fast_pathway_size

        with open(os.path.join(self.imagesets_path, f'{self.subset}.txt'), 'r') as f:
            tmp = f.readlines()
        sequences_names = [x.strip() for x in tmp][1:2]
        self.sequences = defaultdict(dict)
        global_image_id = 0
        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            # masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]['masks'] = masks

        self.data = []  # padding is a tuple like (False,False) first one indicates need to append before the sequence, second one after the sequence
        # create actual data instead of paths, and compute bboxes
        for seq in tqdm(self.sequences, total=len(self.sequences), desc='Preparing Sequences'):
            imgs = []
            masks = []
            boxes = []
            for img, msk in zip(self.sequences[seq]['images'], self.sequences[seq]['masks']):
                '''
                All pre-trained models expect input images normalized in the same way, 
                i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. 
                The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].'''
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

            seq_len = len(imgs)
            start_idx = 0
            for idx in range(ceil(seq_len / self.max_seq_length)):
                batch_imgs = imgs[start_idx:start_idx + self.max_seq_length]
                batch_masks = masks[start_idx:start_idx + self.max_seq_length]
                batch_boxes = boxes[start_idx:start_idx + self.max_seq_length]
                start_idx += self.max_seq_length - self.fast_pathway_size
                padding = (idx == 0, idx + 1 == ceil(seq_len / self.max_seq_length))

                batch_targets = []
                for i in range(len(batch_imgs)):
                    target = {}
                    if len(batch_boxes[i]) == 0:
                        batch_targets.append(target)
                        continue
                    bxs = torch.as_tensor(batch_boxes[i], dtype=torch.float32)
                    target["boxes"] = bxs
                    target["labels"] = torch.ones((len(bxs),), dtype=torch.int64)
                    target["masks"] = torch.as_tensor(batch_masks[i], dtype=torch.uint8)
                    target["image_id"] = torch.tensor([global_image_id])
                    global_image_id += 1
                    target["area"] = (bxs[:, 3] - bxs[:, 1]) * (bxs[:, 2] - bxs[:, 0])
                    target["iscrowd"] = torch.zeros((len(bxs),), dtype=torch.int64)

                    batch_targets.append(target)

                self.data.append((batch_imgs, tuple(batch_targets), padding))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        imgs, targets, padding = self.data[i]
        if self.transforms:
            transformed_imgs = []
            for img in imgs:
                transformed_imgs.append(self.transforms(img))

            return transformed_imgs, targets, padding

        return imgs, targets, padding

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
