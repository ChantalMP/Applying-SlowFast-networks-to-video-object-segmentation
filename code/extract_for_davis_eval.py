from torch.utils.data import DataLoader
from helpers.dataset import DAVISDataset
import torch
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from PIL import Image
from pathlib import Path
from helpers.model import SegmentationModel
from helpers.constants import best_model_path, slow_pathway_size, fast_pathway_size, model_name


def extract_for_davis_evaluation(model):
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS_2016', subset='val', transforms=transforms, year='2016')
    dataloader = DataLoader(dataset, batch_size=None)
    model.eval()

    for seq_idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating Segmentations"):

        imgs, targets, seq_name = seq
        seq_output_path = Path(f'davis2017-evaluation/results/unsupervised/{model_name}/{seq_name}')
        seq_output_path.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            _, detections = model(imgs, deepcopy(targets))

        for img_idx, target in enumerate(targets):
            img = imgs[img_idx].cpu().numpy().transpose(1, 2, 0)
            total_mask = np.zeros(img.shape[:2]).astype(np.bool)

            for mask_pred in detections[img_idx]['masks']:
                mask_pred = (mask_pred.cpu().numpy() >= 0.5)[0]
                total_mask = np.logical_or(total_mask, mask_pred)

            Image.fromarray(total_mask).save(seq_output_path / f'{str(img_idx).zfill(5)}.png')


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                 fast_pathway_size=fast_pathway_size)
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    extract_for_davis_evaluation(model)
