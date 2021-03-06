from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from davis2017_evaluation.davis2017.evaluation import DAVISEvaluation
from helpers.constants import best_model_path, slow_pathway_size, fast_pathway_size
from helpers.constants import model_name
from helpers.dataset import DAVISDataset
from helpers.model import SegmentationModel


def davis_evaluation(model, seq_name_to_process=None):
    transforms = Compose([ToTensor()])
    sequences = 'all' if seq_name_to_process is None else seq_name_to_process
    dataset = DAVISDataset(root='data/DAVIS_2016', subset='val', transforms=transforms, year='2016', sequences=sequences)
    dataloader = DataLoader(dataset, batch_size=None)
    model.eval()
    time_start = time()
    task_type = 'unsupervised' if seq_name_to_process is None else 'semi-supervised'

    for seq_idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating Segmentations"):
        imgs, targets, seq_name = seq
        seq_output_path = Path(f'davis2017_evaluation/results/{task_type}/{model_name}/{seq_name}')
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

    print(f'Evaluating sequences...')
    # Create dataset and evaluate
    # TODO task semi-supervised
    dataset_eval = DAVISEvaluation(davis_root='data/DAVIS_2016', task='unsupervised', gt_set='val', year='2016', sequences=sequences)
    metrics_res = dataset_eval.evaluate(f'davis2017_evaluation/results/{task_type}/{model_name}')
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)

    # Print the results
    print(f"--------------------------- Global results for val ---------------------------\n")
    print(table_g.to_string(index=False))
    print(f"\n---------- Per sequence results for val ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time() - time_start
    print('\nTotal time:' + str(total_time))

    if seq_name_to_process is None:
        return table_g['J&F-Mean'][0], total_time
    else:
        return table_g['J&F-Mean'][0], table_seq['J-Mean'][0], table_seq['F-Mean'][0], total_time


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                 fast_pathway_size=fast_pathway_size)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    model.load_state_dict(torch.load(best_model_path))
    # First do an evaluation to check everything works
    davis_evaluation(model)
