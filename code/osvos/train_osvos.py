from torchvision import transforms
from osvos import osvos_transforms as tr

# Transforms from OSVOS paper:
composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])

# TODO finish dataset (integrate transforms)
# TODO write training loop using dataloader (maybe adapt dataset loading mechanics here)
# TODO Test augmentations on masks and boxes
# TODO load our pretrained network
# TODO evaluation: predict on all other images of sequence
# TODO build whole pipeline that evaluates osvos on all sequences (train + validate)
