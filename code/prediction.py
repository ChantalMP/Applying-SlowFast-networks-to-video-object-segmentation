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
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='train', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load("models/model_overfit_efficientnet.pth"))


    for seq in tqdm(dataloader, total=len(dataloader), desc="Sequence:"):
        preds = []
        imgs, gt_masks, boxes = seq
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            preds.extend(model(imgs, boxes, gt_masks))

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

                print(f'IoU: {intersection_over_union(full_mask.numpy(),full_mask):.4f}')

                full_mask = np.expand_dims(full_mask.numpy().astype(np.float), axis=-1).repeat(4, axis=-1)
                full_mask[:,:,0] = 0.
                full_mask[:,:,1] = 1
                full_mask[:,:,2] = 0.

                mask_idx +=1
                ax.imshow(full_mask, alpha=0.3)

            plt.show()
            a = 1

    pass

if __name__ == '__main__':
    predict_and_visualize()
