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
from helpers.davis_evaluate import davis_evaluation
from torch.utils.tensorboard import SummaryWriter
from helpers.constants import best_model_path, model_path, checkpoint_path, slow_pathway_size, fast_pathway_size, use_proposals, use_rpn_proposals, \
    continue_training

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
    # TODO use predicted boxes in training TOTEST
    # TODO test gpu usage
    # TODO train on davis16 dataset and only use box with the highest prob (probably finetune maskrcnn on davis16 as well)
    # TODO Just reduce score_threshold of current architecture for eval
    # TODO Only feed most probably boxes and use like ground truth (no sorting out etc)
    # TODO if osvos with first picture no context is learned on the left -> overfit on middle frame
    # TODO start with our trained network and finetune it for osvos
    # TODO stride
    # TODO play with threshold for model for evaluation
    # TODO test bigger slowfast
    # TODO avg of N runs

    epochs = 15
    lr = 0.001
    weight_decay = 0.0001

    transforms = Compose([ToTensor()])
    dataset = DAVISDataset(root='data/DAVIS', subset='train', transforms=transforms, use_rpn_proposals=use_rpn_proposals)
    dataloader = DataLoader(dataset, batch_size=None)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model: SegmentationModel = SegmentationModel(device=device, slow_pathway_size=slow_pathway_size,
                                                 fast_pathway_size=fast_pathway_size, use_proposals=use_proposals, use_rpn_proposals=use_rpn_proposals)
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
            total_loss += batch_loss

            global_step += 1

        print(f'\nLoss: {total_loss:.4f}\n')
        writer.add_scalar('Loss/Train', total_loss, global_step=global_step)
        val_iou = davis_evaluation(model)
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
