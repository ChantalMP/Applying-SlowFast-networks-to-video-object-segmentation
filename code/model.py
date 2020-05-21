import torch
from torch import nn
import torchvision.models as models


# TODO consider how to merge slow and fast, also consider regular lateral connections if necessary

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        self.fast_conv1 = nn.Conv3d(
            in_channels=512,
            out_channels=30,
            kernel_size=(16, 3, 3))  # TODO consider padding

        self.slow_conv1 = nn.Conv3d(
            in_channels=512,
            out_channels=30,
            kernel_size=(4, 3, 3)  # TODO consider padding
            # TODO first version is without stride but with smaller kernel
        )

        # TODO lateral connection

        # TODO maskrcnn head
        # TODO might be interesting, from track-rcnn: The 3D convolutions are initialized to an identity function after which the ReLU is applied.
        # TODO 1024 feature maps between the back- bone and the region proposal network
        # TODO in mask rcnn for one image, backbone returns 5(probably not relevant that it is 5) features each with a shape of BATCHSIZEx256xwidthxheight
        # TODO roi heads receive the same features and an input of 2000x4 as proposals(boxes probably)
        # TODO mask roi pool also gets these 5 features but can also but called with just one (multiscaleroialign) also mask_proposals represented by 4 points not sure about the orientation
        # TODO maskrcnn_loss supports discretinzation size (meaning they don't have to be the same size)

        pass

    def extract_resnet_features(self, x):
        # See note [TorchScript super()]
        resnet = self.resnet
        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)

        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        x = resnet.layer4(x)
        return x

    def forward(self, x):
        resnet_features = self.extract_resnet_features(x)
        # TODO this can actually be vectorized, but not sure if it is possible to do all at once do to memory constraints, shouldn't effect the outcome nonetheless

        image_idx = 10
        fast_features = self.fast_conv1(resnet_features[image_idx - 8:image_idx + 8].unsqueeze(0).transpose(1, 2))
        slow_features = self.slow_conv1(resnet_features[image_idx - 2:image_idx + 2].unsqueeze(0).transpose(1, 2))

        print('hi')
        pass


if __name__ == '__main__':
    # TODO consider normalization? Values between 0 and 1?
    image_sequence = torch.rand((20, 3, 256, 256), dtype=torch.float32)
    model = SegmentationModel()
    model(image_sequence)

    pass
# resnet18 = models.mnasnet1_0(pretrained=True)

# Find total parameters and trainable parameters(most paramaters are not trainable, which speed up training a lot)
# total_params = sum(p.numel() for p in resnet18.parameters())
# print(f'{total_params:,} total parameters.')
# total_trainable_params = sum(
#     p.numel() for p in resnet18.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')
