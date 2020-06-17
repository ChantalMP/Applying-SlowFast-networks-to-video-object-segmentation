from torch.utils.data import DataLoader
from helpers.dataset import DAVISDataset
import torch
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from PIL import Image
from pathlib import Path
from helpers.constants import model_name
from davis2017_evaluation.davis2017.evaluation import DAVISEvaluation
import pandas as pd
from time import time


def davis_evaluation(model):
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS_2016', subset='val', transforms=transforms, year='2016')
    dataloader = DataLoader(dataset, batch_size=None)
    model.eval()
    time_start = time()

    for seq_idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating Segmentations"):

        imgs, targets, seq_name = seq
        seq_output_path = Path(f'davis2017_evaluation/results/unsupervised/{model_name}/{seq_name}')
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
    dataset_eval = DAVISEvaluation(davis_root='data/DAVIS_2016', task='unsupervised', gt_set='val', year='2016')
    metrics_res = dataset_eval.evaluate(f'davis2017_evaluation/results/unsupervised/{model_name}')
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

    return table_g['J&F-Mean'][0], total_time
