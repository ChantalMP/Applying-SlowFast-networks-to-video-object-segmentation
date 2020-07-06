from collections import defaultdict

import numpy as np
import os
import random
import torch

from helpers.constants import best_model_path, slow_pathway_size, fast_pathway_size, random_seed
from helpers.davis_evaluate import davis_evaluation

# As deterministic as possible
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from osvos.dataset_osvos import DAVISSequenceDataset
from osvos.osvos_model import OsvosSegmentationModel
from helpers.model import SegmentationModel


def evaluate_model(model, device, sequence_name):
    full_model = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size, fast_pathway_size=fast_pathway_size)
    full_model.load_state_dict(model.state_dict())
    full_model.to(device)
    model.to(torch.device('cpu'))
    jf_mean, j_mean, f_mean, total_time = davis_evaluation(full_model, seq_name_to_process=sequence_name)
    model.to(device)
    del full_model
    return jf_mean, j_mean, f_mean, total_time


def main(sequence_name):
    cfg = object
    epochs = cfg.epochs
    lr = cfg.lr
    weight_decay = 0.0001

    transforms = Compose([ToTensor()])
    dataset = DAVISSequenceDataset(root='data/DAVIS_2016', transforms=transforms, sequence_name=sequence_name, fast_pathway_size=fast_pathway_size)
    dataloader = DataLoader(dataset, batch_size=None)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: OsvosSegmentationModel = OsvosSegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                           fast_pathway_size=fast_pathway_size)
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    global_step = 0
    all_results = defaultdict(dict)

    # First do an evaluation to check everything works
    # TODO evaluate model can remain the same all the time?
    jf_mean, j_mean, f_mean, total_time = evaluate_model(model=model, device=device, sequence_name=sequence_name)
    all_results[-1] = {'jfmean': jf_mean, 'jmean': j_mean, 'fmean': f_mean, 'eval_time': total_time}
    for epoch in tqdm(range(0, epochs), total=epochs, desc="Epochs"):
        total_loss = 0.
        for idx, seq in enumerate(dataloader):
            model.train()
            imgs, target = seq
            batch_loss, _ = model(imgs, target, optimizer=opt)  # Backward happens inside
            total_loss += batch_loss

            global_step += 1

        print(f'\nLoss: {total_loss:.4f}\n')

        jf_mean, j_mean, f_mean, total_time = evaluate_model(model=model, device=device, sequence_name=sequence_name)
        all_results[epoch] = {'jfmean': jf_mean, 'jmean': j_mean, 'fmean': f_mean, 'eval_time': total_time}

        save_path = best_model_path.parent / f'{best_model_path.name.replace(".pth", "")}_osvos_{sequence_name}.pth'
        torch.save(model.state_dict(), save_path)

    print("Finished Training.")
    return all_results


if __name__ == '__main__':
    main(sequence_name='bmx-trees')
