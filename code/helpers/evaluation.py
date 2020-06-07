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
from helpers.constants import eval_output_path, model_name, pred_output_path


def evaluate(model, writer=None, global_step=None, save_all_imgs=False):
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='val', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=None)
    model.eval()

    intersection_over_unions = []
    for seq_idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating with Sequence:"):
        plotted = False
        plt_needed = False

        imgs, targets, seq_name = seq
        with torch.no_grad():
            _, detections = model(imgs, deepcopy(targets))

        for img_idx, target in enumerate(targets):
            img = imgs[img_idx].cpu().numpy().transpose(1, 2, 0)
            if not plotted:
                ax = plt.subplot(1, 1, 1)
                ax.set_axis_off()
                ax.imshow(img)
                ax.axis('off')

                plt_needed = True

            if len(target) == 0:
                continue

            total_gt_mask = np.zeros(img.shape[:2]).astype(np.bool)
            for gt_mask in target['masks']:
                gt_mask = (gt_mask.cpu().numpy() >= 0.5)
                total_gt_mask = np.logical_or(total_gt_mask, gt_mask)

            total_pred_mask = np.zeros(img.shape[:2]).astype(np.bool)
            for mask_pred in detections[img_idx]['masks']:
                mask_pred = (mask_pred.cpu().numpy() >= 0.5)[0]
                total_pred_mask = np.logical_or(total_pred_mask, mask_pred)

            iou = intersection_over_union(total_gt_mask, total_pred_mask)
            intersection_over_unions.append(iou)

            if not plotted:
                full_mask = np.expand_dims(total_pred_mask, axis=-1).repeat(4, axis=-1).astype(np.float)
                full_mask[:, :, 0] = 0.
                full_mask[:, :, 1] = 1
                full_mask[:, :, 2] = 0.

                ax.imshow(full_mask, alpha=0.3)
                if not save_all_imgs:
                    plotted = True

            if plt_needed:
                if save_all_imgs:
                    output_path = pred_output_path / model_name / seq_name
                    output_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_path / f'{seq_name}_{img_idx}.png', bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(eval_output_path / f'{seq_name}_{img_idx}.png', bbox_inches='tight', pad_inches=0)
                plt.clf()
                plt_needed = False

    mean_iou = statistics.mean(intersection_over_unions)

    print(f'\nMean_IoU: {mean_iou:.4f}\n')

    if writer is not None and global_step is not None:
        writer.add_scalar('IoU/Mean', mean_iou, global_step=global_step)

    return mean_iou