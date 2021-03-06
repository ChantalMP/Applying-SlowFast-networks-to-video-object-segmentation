# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate, predict_boxes
import utils
import transforms as T
from torchvision.transforms import ToTensor
from data_aug import data_aug


class DavisDataset(object):
    def __init__(self, root, transforms, year='2017', subset='val', augmentation=False):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(Path(os.path.join(self.root, 'JPEGImages', '480p')).glob('*/*.jpg')))
        self.masks = list(sorted(Path(os.path.join(self.root, 'Annotations', '480p')).glob('*/*.png')))
        self.imagesets_path = os.path.join(self.root, 'ImageSets', '2017') if year == '2017' else os.path.join(
            self.root, 'ImageSets', '480p')
        with open(os.path.join(self.imagesets_path, f'train.txt'), 'r') as f:
            tmp = f.readlines()
            train_img_names = [x.strip() for x in tmp] if year == '2017' else sorted(
                {x.split()[0].split('/')[-2] for x in tmp})
            train_img_names = set(train_img_names)
        with open(os.path.join(self.imagesets_path, f'val.txt'), 'r') as f:
            tmp = f.readlines()
            val_img_names = [x.strip() for x in tmp] if year == '2017' else sorted(
                {x.split()[0].split('/')[-2] for x in tmp})
            val_img_names = set(val_img_names)

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        for idx, img_path in enumerate(self.imgs):
            img_name = img_path.parent.name
            if img_name in train_img_names:
                self.train_indices.append(idx)
            elif img_name in val_img_names:
                self.val_indices.append(idx)
            else:
                self.test_indices.append(idx)

        self.totensor = ToTensor()
        self.subset = subset
        self.augmentation = augmentation
        self.random_horizontal_flip = data_aug.RandomHorizontalFlip()
        self.scale = data_aug.RandomScale(scale=0.15)
        self.rotate = data_aug.RandomRotate(angle=10)

    def apply_augmentations(self, imgs, masks, gt_boxes):
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

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        seq_name = img_path.parts[-2]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        valid_ids = []
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin < xmax and ymin < ymax:
                boxes.append([xmin, ymin, xmax, ymax])
                valid_ids.append(i)

        num_objs = len(valid_ids)
        masks = masks[valid_ids]

        if self.subset == 'train' and self.augmentation:  # We are using a version without augmentations for now
            img, masks, boxes = self.apply_augmentations([np.array(img)], [masks], [boxes])
            img = img[0]
            masks = masks[0]
            if len(masks) > 0:
                masks = np.stack(masks)
            boxes = boxes[0]

        if len(boxes) == 0:
            if self.transforms is not None:
                img = T.ToTensor()(img, None)[0]
            return img, {}, False, seq_name
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.augmentation:
            img = self.totensor(img)
        elif self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, True, seq_name

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    # load a pre-trained model for classification and return only the features
    # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    #
    # # MaskRCNN needs to know the number of output channels in a backbone. For mobilenet_v2, it's 1280, so we need to add it here
    # backbone.out_channels = 1280
    #
    # # let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios. We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # # let's define what are the feature maps that we will use to perform the region of interest cropping, as well as the size of the crop after rescaling. if your backbone returns a Tensor, featmap_names is expected to be [0]. More generally, the backbone should return an OrderedDict[Tensor], and in featmap_names you can choose which feature maps to use.
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,
    #                                                 sampling_ratio=2)
    #
    # mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=14,sampling_ratio=2)
    #
    # model = MaskRCNN(backbone,
    #                  num_classes=2,
    #                  rpn_anchor_generator=anchor_generator,
    #                  box_roi_pool=roi_pooler,
    #                  mask_roi_pool=mask_roi_pooler)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(train=True, year=None, split=None):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    if train:
        dataset = DavisDataset('../data/DAVIS', get_transform(train=True), subset='train', augmentation=False)
        dataset_val = DavisDataset('../data/DAVIS', get_transform(train=False))
    else:  # box computation
        dataset = DavisDataset('../data/DAVIS', get_transform(train=False))
        if year is None or year == '2017':
            dataset_val = DavisDataset('../data/DAVIS', get_transform(train=False), year='2017')
        else:
            dataset_val = DavisDataset('../data/DAVIS_2016', get_transform(train=False), year='2016')

    # split the dataset in train and test set
    dataset = torch.utils.data.Subset(dataset, dataset.train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val, dataset_val.val_indices)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    if not train:
        model.load_state_dict(torch.load('maskrcnn_model.pth'))

    # move model to the right device
    model.to(device)

    if train:
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's train it for 10 epochs
        num_epochs = 15

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_val, device=device)
            torch.save(model.state_dict(), "data/maskrcnn_model_augmented.pth")

        print("That's it!")

    else:
        if split is None or split == 'train':
            predict_boxes(model, data_loader, device=device, year=year, split=split)
        else:
            predict_boxes(model, data_loader_val, device=device, year=year, split=split)


if __name__ == "__main__":
    main(train=True, year='2017', split='train')
