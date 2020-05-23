# TODO
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


def main():
    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='train', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model : SegmentationModel= SegmentationModel(device=device)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
    epochs = 20

    for epoch in tqdm(range(epochs), total=epochs, desc="Epoch:"):
        for seq in tqdm(dataloader, total=len(dataloader), desc="Sequence:"):
            imgs, masks, boxes = seq
            imgs = torch.stack(imgs)[:, 0, :, :].to(device)  # TODO decide if 0 should be first or second
            loss = model(imgs, boxes, masks)
            print(loss)
            loss.backward()
            opt.step()
            opt.zero_grad()

    pass


if __name__ == '__main__':
    main()
