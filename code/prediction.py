from torch.utils.data import DataLoader
from dataset import DAVISDataset
from model import SegmentationModel
import torch
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from utils import convert_mask_pred_to_ground_truth_format,intersection_over_union

def predict_and_visualize():
    slow_pathway_size = 4
    fast_pathway_size = 16
    overlap = fast_pathway_size // 2
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='val', transforms=transforms, max_seq_length=200,
                           fast_pathway_size=fast_pathway_size)
    dataloader = DataLoader(dataset, batch_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                 fast_pathway_size=fast_pathway_size)
    model.to(device)
    model.load_state_dict(torch.load("models/model_best_onecycle_scheduler_resnet18_bn.pth"))

    for idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Sequence:"):
        model.eval()
        preds = []
        imgs, gt_masks, boxes, padding = seq
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            _, output = model(imgs, boxes, gt_masks, padding)
            preds.extend(output)

        # imgs can contain padding values not predicted by the model, delete them
        if not padding[0].item():
            imgs = imgs[overlap:]
            gt_masks = gt_masks[overlap:]
            boxes = boxes[overlap:]
        if not padding[1].item():
            imgs = imgs[:-overlap]
            gt_masks = gt_masks[:-overlap]
            boxes = boxes[:-overlap]

        mask_idx = 0
        for img_idx, (img_boxes, img_gt_masks) in enumerate(zip(boxes, gt_masks)):
            img = imgs[img_idx].cpu().numpy().transpose(1,2,0)
            ax = plt.subplot(1, 1, 1)
            ax.set_axis_off()
            ax.imshow(img)
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

            plt.savefig(f'data/output/pred_output/{mask_idx}.png')


if __name__ == '__main__':
    predict_and_visualize()
