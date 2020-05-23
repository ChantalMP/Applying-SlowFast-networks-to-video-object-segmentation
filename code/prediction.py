from torch.utils.data import DataLoader
from dataset import DAVISDataset
from model import SegmentationModel
import torch
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from matplotlib import pyplot as plt


def predict():
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='train', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load("models/model_overfit.pth"))

    preds = []
    for seq in tqdm(dataloader, total=len(dataloader), desc="Sequence:"):
        imgs, masks, boxes = seq
        imgs = torch.stack(imgs)[:, 0, :, :].to(device)  # TODO decide if 0 should be first or second
        with torch.no_grad():
            preds.extend(model(imgs, boxes, masks))

        for pred in preds:
            plt.imshow()

    pass

if __name__ == '__main__':
    predict()
