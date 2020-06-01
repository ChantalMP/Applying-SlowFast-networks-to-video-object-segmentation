from torch.utils.data import DataLoader
import statistics
from dataset import DAVISDataset
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from helpers.utils import convert_mask_pred_to_ground_truth_format, intersection_over_union, revert_normalization
from copy import deepcopy


def evaluate(model, device, writer=None, global_step=None):
    # overlap = model.fast_pathway_size // 2
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='val', transforms=transforms, max_seq_length=50,
                           fast_pathway_size=16)
    dataloader = DataLoader(dataset, batch_size=None)
    model.eval()

    intersection_over_unions = []
    for seq_idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating with Sequence:"):
        plotted = False

        imgs, targets, padding = seq
        with torch.no_grad():
            _, detections = model(imgs, deepcopy(targets), padding)
        # count += imgs.shape[0] - (int(padding[0]) * overlap) - (int(padding[1]) * overlap)
        # with torch.no_grad():
        #     loss, output = model(imgs, boxes, gt_masks, padding)
        #     total_loss += loss.item()
        #     preds.extend(output)

        # imgs can contain padding values not predicted by the model, delete them
        # if not padding[0].item():
        #     imgs = imgs[overlap:]
        #     gt_masks = gt_masks[overlap:]
        #     boxes = boxes[overlap:]
        # if not padding[1].item():
        #     imgs = imgs[:-overlap]
        #     gt_masks = gt_masks[:-overlap]
        #     boxes = boxes[:-overlap]

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
                        plotted = True

            if plt_needed:
                plt.savefig(f'data/output/eval_output/{seq_idx}_{img_idx}.png')
                plt.clf()
                plt_needed = False

    mean_iou = statistics.mean(intersection_over_unions)
    median_iou = statistics.median(intersection_over_unions)

    print(f'\nMean_IoU: {mean_iou:.4f}\n'
          f'Median_IoU: {median_iou:.4f}\n')

    # if writer is not None and global_step is not None:
    #     writer.add_scalar('Loss/Val', total_loss, global_step=global_step)
    #     writer.add_scalar('IoU/Mean', mean_iou, global_step=global_step)
    #     writer.add_scalar('IoU/Median', median_iou, global_step=global_step)

    return mean_iou
