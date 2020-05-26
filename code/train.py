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
from dataset import DAVISDataset
from torch.utils.data import DataLoader
from model import SegmentationModel
import torch
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from helpers.evaluation import evaluate
from torch.utils.tensorboard import SummaryWriter


def main():
    epochs = 2000
    lr = 1e-4
    logging_frequency = 50
    slow_pathway_size = 4
    fast_pathway_size = 16

    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='train', transforms=transforms, max_seq_length=60,
                           fast_pathway_size=fast_pathway_size)
    dataloader = DataLoader(dataset, batch_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                 fast_pathway_size=fast_pathway_size)
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    opt = torch.optim.AdamW(params=model.parameters(), lr=lr)
    writer: SummaryWriter = SummaryWriter()
    global_step = 0
    best_iou = -1

    for epoch in tqdm(range(epochs), total=epochs, desc="Epoch:"):
        total_loss = 0.
        count = 0
        for idx, seq in tqdm(enumerate(dataloader), total=len(dataloader), desc="Sequence:"):
            model.train()
            imgs, gt_masks, boxes, padding = seq
            imgs = torch.cat(imgs).to(device)
            loss = model(imgs, boxes, gt_masks, padding)
            total_loss += loss.item()
            count += imgs.shape[0] - (int(padding[0]) * 8) - (int(padding[1]) * 8)
            loss.backward()
            opt.step()
            opt.zero_grad()
            global_step += 1

            if idx % logging_frequency == 0:
                total_loss = total_loss / count
                print(f'\nLoss: {total_loss:.4f}\n')
                writer.add_scalar('Loss/Train', total_loss, global_step=global_step)
                total_loss = 0.
                count = 0

                val_iou = evaluate(model, device, writer=writer, global_step=global_step)

                if val_iou > best_iou:
                    best_iou = val_iou
                    print(f'Saving model with iou: {val_iou}')
                    torch.save(model.state_dict(), "models/model_best.pth")

    torch.save(model.state_dict(), "models/model.pth")


if __name__ == '__main__':
    main()
