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

                if not plotted:
                    full_mask = np.expand_dims(mask, axis=-1).repeat(4, axis=-1)
                    full_mask[:, :, 0] = 0.
                    full_mask[:, :, 1] = 1
                    full_mask[:, :, 2] = 0.

                    ax.imshow(full_mask, alpha=0.3)
                    plotted_count += 1
                    if plotted_count == len(target['masks']):
                        if not save_all_imgs:
                            plotted = True

            if plt_needed:
                if save_all_imgs:
                    output_path = pred_output_path / model_name / seq_name
                    output_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_path / f'{seq_name}_{img_idx}.png')
                else:
                    plt.savefig(eval_output_path / f'{seq_name}_{img_idx}.png')
                plt.clf()
                plt_needed = False

    mean_iou = statistics.mean(intersection_over_unions)

    print(f'\nMean_IoU: {mean_iou:.4f}\n')

    if writer is not None and global_step is not None:
        writer.add_scalar('IoU/Mean', mean_iou, global_step=global_step)

    return mean_iou