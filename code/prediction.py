from torch.utils.data import DataLoader
from helpers.dataset import DAVISDataset
from helpers.model import SegmentationModel
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from helpers.utils import intersection_over_union, convert_mask_pred_to_ground_truth_format, revert_normalization
from helpers.constants import best_model_path, pred_output_path

def predict_and_visualize():
    slow_pathway_size = 4
    fast_pathway_size = 4
    overlap = fast_pathway_size // 2
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transforms = Compose([ToTensor(), Normalize(mean=means,
                                                std=stds)])
    dataset = DAVISDataset(root='data/DAVIS', subset='train', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=None)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                 fast_pathway_size=fast_pathway_size)
    model.to(device)
    model.load_state_dict(torch.load(best_model_path))

    for idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Sequence:"):
        model.eval()
        preds = []
        imgs, gt_masks, boxes = seq
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            _, output = model(imgs, boxes, gt_masks)
            preds.extend(output)

        mask_idx = 0
        for img_idx, (img_boxes, img_gt_masks) in enumerate(zip(boxes, gt_masks)):
            img = imgs[img_idx].cpu().numpy().transpose(1,2,0)
            ax = plt.subplot(1, 1, 1)
            ax.set_axis_off()
            ax.imshow(revert_normalization(img, means=means, stds=stds))
            for box, gt_mask in zip(img_boxes, img_gt_masks): # Wont work when not using gt_boxes because we can have less boxes than masks
                box = box[0].tolist()
                mask = preds[mask_idx].cpu().numpy().astype(np.float)
                full_mask = convert_mask_pred_to_ground_truth_format(img=img,box=box,mask=mask,threshold=0.5)

                print(f'Idx-IoU: {mask_idx} - {intersection_over_union(gt_mask[0].numpy(), full_mask):.4f}')

                full_mask = np.expand_dims(full_mask, axis=-1).repeat(4, axis=-1)
                full_mask[:,:,0] = 0.
                full_mask[:,:,1] = 1
                full_mask[:,:,2] = 0.

                mask_idx +=1
                ax.imshow(full_mask, alpha=0.3)

            plt.savefig(pred_output_path / f'{mask_idx}.png')


if __name__ == '__main__':
    predict_and_visualize()
