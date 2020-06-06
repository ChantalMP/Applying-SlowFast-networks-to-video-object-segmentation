# Dataset and Dataloader (support video) for Davis 2017 (compute bounding boxes from mask data)
# Dataloader returns one video as sequences of frames with corresponding masks (resize frames to a certain dimension (maybe 224 or 256)) decide on size
# Model Pipeline: model is called with a sequence of frames(complete video).
# 1. Compute resnet features for all frames (possible optimization, maybe computing for all of them is not computationally possible)
# 2. Split feature maps into subsequences with stride 1
# 3. Use 3D convs in two tracks like SlowFast to compute one final feature map (merge + temporal pooling) for one subsequence
# 4. Use (ground truth) bounding boxes + maskrcnn type of head to compute mask for middle frame

# Training Loop
# Call model pipeline to get mask predictions
# Backprop (eventuall needs clever optimizations), between ground truth masks and all the masks outputted by the model pipeline
# Testing Loop
# Call model to get masks, eventually use offline bounding box computer


# Bounding Box Computation
# Use Faster-RCNN (pretrained on COCO) to get bounding boxes for whole davis dataset

# Evaluation
# Slow rate of fast how is it etc etc.
from helpers.dataset import DAVISDataset
from torch.utils.data import DataLoader
from helpers.model import SegmentationModel
import torch
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from helpers.evaluation import evaluate
from torch.utils.tensorboard import SummaryWriter
from helpers.constants import best_model_path, model_path, slow_pathway_size, fast_pathway_size

'''
New architecture proposal:
Fully train a maskrcnn for davis
Use its transformations and backbone to extract fpn features for every image and save it
Enchance features with temporal context
Use its heads but this time with enchaned features
'''


def main():
    '''
    Train till convergence:
    1. Without temporal context
    2. With but slow fast same size (as big as it fits)
    3. Smaller slow but bigger fast
    '''
    # TODO also try to use predicted boxes instead of ground truth boxes
    # TODO save optimizer
    # TODO evaluate 1 1 model
    # TODO evaluate current 8 8 version
    # TODO presentation
    epochs = 10
    lr = 0.001
    weight_decay = 0.0001

    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='train', transforms=transforms)
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

    writer: SummaryWriter = SummaryWriter()
    global_step = 0
    best_iou = -1

    for epoch in tqdm(range(epochs), total=epochs, desc="Epochs"):
        total_loss = 0.
        for idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Sequences"):
            model.train()
            imgs, targets, _ = seq
            batch_loss, _ = model(imgs, targets, optimizer=opt)  # Backward happens inside
            total_loss += batch_loss

            global_step += 1

        print(f'\nLoss: {total_loss:.4f}\n')
        writer.add_scalar('Loss/Train', total_loss, global_step=global_step)
        val_iou = evaluate(model, writer=writer, global_step=global_step)
        if val_iou > best_iou:
            best_iou = val_iou
            print(f'Saving model with iou: {val_iou}')
            torch.save(model.state_dict(), best_model_path)

    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
