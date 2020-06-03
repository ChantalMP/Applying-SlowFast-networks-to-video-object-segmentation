from torch.utils.data import DataLoader
import statistics
from helpers.dataset import DAVISDataset
import torch
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from helpers.utils import intersection_over_union
from copy import deepcopy
from helpers.constants import eval_output_path


def evaluate(model, writer=None, global_step=None):
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='val', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=None)
    model.eval()

    with open('data/DAVIS/ImageSets/2017/hard_val.txt', 'r') as f:
        hard_sequences = set(f.read().splitlines())

    intersection_over_unions = []
    intersection_over_unions_hard = []
    for seq_idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating with Sequence:"):
        plotted = False

        imgs, targets, seq_name = seq
        with torch.no_grad():
            _, detections = model(imgs, deepcopy(targets))

        plt_needed = False
        for img_idx, target in enumerate(targets):
            img = imgs[img_idx].cpu().numpy().transpose(1, 2, 0)
            if not plotted:
                ax = plt.subplot(1, 1, 1)
                ax.set_axis_off()
                ax.imshow(img)
                plotted_count = 0
                plt_needed = True

            if len(target) == 0:
                continue
            for gt_idx, gt_mask in enumerate(target['masks']):  # Wont work when not using gt_boxes because we can have less boxes than masks
                try:
                    mask = detections[img_idx]['masks'][gt_idx].cpu().numpy().astype(np.float)
                except Exception as e:
                    print('Predicted too few masks')
                    break
                mask = (mask >= 0.5).astype(np.float)[0]
                iou = intersection_over_union(gt_mask.cpu().numpy(), mask)
                intersection_over_unions.append(iou)
                if seq_name in hard_sequences:
                    intersection_over_unions_hard.append(iou)

                if not plotted:
                    full_mask = np.expand_dims(mask, axis=-1).repeat(4, axis=-1)
                    full_mask[:, :, 0] = 0.
                    full_mask[:, :, 1] = 1
                    full_mask[:, :, 2] = 0.

                    ax.imshow(full_mask, alpha=0.3)
                    plotted_count += 1
                    if plotted_count == len(target['masks']):
                        plotted = True

            if plt_needed:
                plt.savefig(eval_output_path / f'{seq_idx}_{img_idx}.png')
                plt.clf()
                plt_needed = False

    mean_iou = statistics.mean(intersection_over_unions)
    mean_iou_hard = statistics.mean(intersection_over_unions_hard)

    print(f'\nMean_IoU: {mean_iou:.4f}\n'
          f'Mean_Hard_IoU: {mean_iou_hard:.4f}\n')

    if writer is not None and global_step is not None:
        writer.add_scalar('IoU/Mean', mean_iou, global_step=global_step)
        writer.add_scalar('IoU/Mean_Hard', mean_iou_hard, global_step=global_step)

    return mean_iou
