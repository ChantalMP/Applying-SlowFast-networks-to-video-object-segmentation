from torchvision import transforms
from osvos import osvos_transforms as tr

# Transforms from OSVOS paper:
# composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
#                                           tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
#                                           tr.ToTensor()])

# TODO finish dataset (integrate transforms)
# TODO write training loop using dataloader (maybe adapt dataset loading mechanics here)
# TODO Test augmentations on masks and boxes
# TODO load our pretrained network
# TODO evaluation: predict on all other images of sequence
# TODO build whole pipeline that evaluates osvos on all sequences (train + validate)
# TODO new model.py code


from helpers.constants import best_model_path, model_path, checkpoint_path, slow_pathway_size, fast_pathway_size, random_seed
import random
import torch
import numpy as np
import os

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
from torch.utils.tensorboard import SummaryWriter
from osvos.dataset_osvos import DAVISSequenceDataset


# from helpers.dataset import DAVISDataset
# from helpers.model import SegmentationModel
# from helpers.davis_evaluate import davis_evaluation


def main():
    '''
    Train till convergence:
    1. Without temporal context
    2. With but slow fast same size (as big as it fits)
    3. Smaller slow but bigger fast
    '''
    # TODO test gpu usage
    # TODO start with our trained network and finetune it for osvos
    # TODO test writer working correctly (also for colab)
    # TODO test eval time working correctly
    # TODO test fixing working correctly
    # TODO maskrcnn augmentation
    epochs = 20
    lr = 0.001
    weight_decay = 0.0001

    transforms = Compose([ToTensor()])
    dataset = DAVISSequenceDataset(root='data/DAVIS_2016', subset='train', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=None)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                 fast_pathway_size=fast_pathway_size)
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    total_steps = epochs * len(dataloader)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    writer: SummaryWriter = SummaryWriter(writer_dir)
    global_step = 0
    best_iou = -1

    if continue_training:
        print('Continuing training')
        checkpoint = torch.load(checkpoint_path)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(torch.load(model_path))
        epoch = checkpoint['epoch'] + 1
    else:
        epoch = 0

    # First do an evaluation to check everything works
    davis_evaluation(model)
    for epoch in tqdm(range(epoch, epochs), total=epochs - epoch, desc="Epochs"):
        total_loss = 0.
        for idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Sequences"):
            model.train()
            imgs, targets, _ = seq
            batch_loss, _ = model(imgs, targets, optimizer=opt)  # Backward happens inside
            writer.add_scalar('Batch Loss/Train', batch_loss, global_step=global_step)
            total_loss += batch_loss

            global_step += 1

        print(f'\nLoss: {total_loss:.4f}\n')
        writer.add_scalar('Loss/Train', total_loss, global_step=global_step)
        val_iou, eval_time = davis_evaluation(model)
        writer.add_scalar('Eval Time', eval_time, global_step=global_step)
        if val_iou > best_iou:
            best_iou = val_iou
            print(f'Saving model with iou: {val_iou}')
            torch.save(model.state_dict(), best_model_path)

        torch.save(model.state_dict(), model_path)
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': opt.state_dict()
        }, checkpoint_path)


if __name__ == '__main__':
    main()
