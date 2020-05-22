import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import ToTensor
from roi_heads_custom import RoIHeads
from torchvision.ops import RoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor


# TODO consider how to merge slow and fast, also consider regular lateral connections if necessary


class SegmentationModel(nn.Module):
    def __init__(self, device):
        super(SegmentationModel, self).__init__()
        self.device = device

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.layer3._modules['0'].conv1.stride = (1, 1)
        self.resnet.layer3._modules['0'].downsample._modules['0'].stride = (1, 1)  # Fixing too much pooling
        self.resnet.layer4._modules['0'].conv1.stride = (1, 1)
        self.resnet.layer4._modules['0'].downsample._modules['0'].stride = (1, 1)  # Fixing too much pooling

        self.fast_conv1 = nn.Conv3d(
            in_channels=512,
            out_channels=64,
            kernel_size=(16, 3, 3))  # TODO consider padding

        self.slow_conv1 = nn.Conv3d(
            in_channels=512,
            out_channels=256,
            kernel_size=(4, 3, 3)  # TODO consider padding
            # TODO first version is without stride but with smaller kernel
        )

        mask_roi_pool = RoIAlign(output_size=14, spatial_scale=1, sampling_ratio=2)  # TODO calculate final value for spatial scale and use that one

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(320, mask_layers, mask_dilation)  # TODo check sizes

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                           mask_dim_reduced, num_classes=1)  # TODO either 1 or 2

        self.roi_head = RoIHeads(mask_roi_pool=mask_roi_pool, mask_head=mask_head, mask_predictor=mask_predictor)

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

    def forward(self, x, bboxes, targets=None):
        resnet_features = self.extract_resnet_features(x)  # TODO extend features of resnet 0s
        resnet_features = torch.cat([torch.zeros_like(resnet_features[:8, :, :, :]), resnet_features, torch.zeros_like(resnet_features[:8, :, :, :])])
        # resnet_features =
        # TODO this can actually be vectorized, but not sure if it is possible to do all at once do to memory constraints, shouldn't effect the outcome nonetheless
        all_features = []

        for idx in range(8, len(resnet_features) - 7):
            # TODO modify this according to slowfast
            fast_features = self.fast_conv1(resnet_features[idx - 8:idx + 8].unsqueeze(0).transpose(1, 2))
            slow_features = self.slow_conv1(resnet_features[idx - 2:idx + 2].unsqueeze(0).transpose(1, 2))
            features = torch.cat([slow_features, fast_features], dim=1)[:, :, 0, :, :]
            all_features.append(features)

        all_features = torch.cat(all_features)
        image_sizes = [tuple(x.shape[2:4])] * len(all_features)
        output = self.roi_head(all_features[:4], bboxes[:4], image_sizes[:4], targets[:4])
        # TODO now like maskrcnn
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
