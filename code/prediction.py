from torch.utils.data import DataLoader
from dataset import DAVISDataset
from model import SegmentationModel
import torch
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


def predict():
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='train', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load("models/model_overfit.pth"))


    for seq in tqdm(dataloader, total=len(dataloader), desc="Sequence:"):
        preds = []
        imgs, masks, boxes = seq
        imgs = torch.cat(imgs).to(device)
        with torch.no_grad():
            preds.extend(model(imgs, boxes, masks))

        mask_idx = 0
        for img_idx, img_boxes in enumerate(boxes):
            img = imgs[img_idx]
            ax = plt.subplot(1, 1, 1)
            ax.set_axis_off()
            ax.set_xlim(0,img.shape[2])
            ax.set_ylim(img.shape[1],0)
            ax.imshow(img.cpu().numpy().transpose(1,2,0))
            for box in img_boxes:
                box = box[0].tolist()
                mask = preds[mask_idx].cpu().numpy().astype(np.float)

                # resize mask
                mask_plot = Image.fromarray(mask)
                mask_plot = np.array(mask_plot.resize((256, 256), resample=Image.ANTIALIAS))
                #theshold mask
                mask_plot = (mask_plot >= 0.5).astype(float)

                mask_plot = np.expand_dims(mask_plot, axis=-1).repeat(4, axis=-1)
                mask_plot[:,:,0] = 0.
                mask_plot[:,:,1] = 1
                mask_plot[:,:,2] = 0

                mask_idx +=1
                ax.imshow(mask_plot, alpha=0.3, extent=(box[0], box[2], box[3], box[1]), interpolation='antialiased')
            plt.show()

    pass

if __name__ == '__main__':
    predict()
