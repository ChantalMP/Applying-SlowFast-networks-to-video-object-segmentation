import math
import sys
import time
import torch


import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets, valids, _ in metric_logger.log_every(data_loader, print_freq, header):
        if False in valids:
            continue
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def convert_mask_pred_to_ground_truth_format(img, box, mask, threshold=0.5):
    # resize mask
    resized_mask = Image.fromarray(mask)
    width, height = box[2] - box[0], box[3] - box[1]
    resized_mask = np.array(resized_mask.resize((width, height), resample=Image.ANTIALIAS))
    # Fill a image sizes mask with the values of resized mask at the corresponding location
    full_mask = np.zeros_like(img[:, :, 0])
    full_mask[box[1]:box[3], box[0]:box[2]] = resized_mask

    # threshold mask
    full_mask = (full_mask >= threshold).astype(np.float)

    return full_mask

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    count = 0
    for images, targets, valids, _ in metric_logger.log_every(data_loader, 100, header):
        if False in valids:
            continue
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        print("call model")
        outputs = model(images)
        print("calculated outputs")

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        img = images[0].cpu().numpy().transpose(1, 2, 0)
        ax = plt.subplot(1, 1, 1)
        ax.set_axis_off()
        ax.imshow(img)
        count += 1
        mask_count = 3
        for box, mask_tensor in zip(outputs[0]['boxes'], outputs[0]['masks']):  # Wont work when not using gt_boxes because we can have less boxes than masks
            box = box.int().tolist()
            mask = ((mask_tensor[0].cpu().numpy().astype(np.float)) >= 0.5).astype(np.float)

            full_mask = np.expand_dims(mask, axis=-1).repeat(4, axis=-1)
            full_mask[:, :, 0] = 0.
            full_mask[:, :, 1] = 1
            full_mask[:, :, 2] = 0.

            ax.imshow(full_mask, alpha=0.3)
            mask_count -= 1
            if mask_count == 0:
                break

        plt.savefig(f'data/{count}.png')
        plt.clf()

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.no_grad()
def predict_boxes(model, data_loader, device):
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    all_boxes = defaultdict(list)

    for images, targets, valids, seq_names in tqdm(data_loader, total=len(data_loader)):

        plotted = False
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for i in range(len(seq_names)):
            all_boxes[seq_names[i]].append(outputs[i]['boxes'].cpu())

        # Plot boxes
        # for img in images:
        #     img = img.cpu().numpy().transpose(1, 2, 0)
        #     ax = plt.subplot(1, 1, 1)
        #     ax.set_axis_off()
        #     ax.imshow(img)
        #     for box, mask_tensor in zip(outputs[0]['boxes'], outputs[0]['masks']):  # Wont work when not using gt_boxes because we can have less boxes than masks
        #         box = box.int().tolist()
        #
        #         x = box[0]
        #         y = box[1]
        #         width = box[2] - x
        #         height = box[3] - y
        #         rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        #
        #         # Add the patch to the Axes
        #         ax.add_patch(rect)
        #
        #     if not plotted:
        #         plt.show()
    torch.save(all_boxes, "predicted_boxes_train.pt")
