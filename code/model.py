from math import ceil
import torch
from torch import nn
from roi_heads_custom import RoIHeads
from torchvision.ops import RoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from efficientnet_pytorch import EfficientNet
from torchvision import models


# TODO consider how to merge slow and fast, also consider regular lateral connections if necessary

class FeatureExtractor(nn.Module):
    def __init__(self, name='resnet_50'):
        super(FeatureExtractor, self).__init__()
        supported_extractors = ['resnet_18', 'resnet_50', 'efficientnet-b0']
        if name not in supported_extractors:
            print(f'{name} not supported')

        else:
            print(f'{name} feature extractor')

        self.name = name

        if self.name == 'resnet_18':
            self.net = models.resnet18(pretrained=True)
            self.net.layer3._modules['0'].conv1.stride = (1, 1)
            self.net.layer3._modules['0'].downsample._modules['0'].stride = (1, 1)
            self.net.layer4._modules['0'].conv1.stride = (1, 1)
            self.net.layer4._modules['0'].downsample._modules['0'].stride = (1, 1)
            self.output_size = 512

        elif self.name == 'resnet_50':
            self.net = models.resnet50(pretrained=True)
            self.output_size = 512
            # Either make it not get that small or take from earlier layers
            # self.net.layer3._modules['0'].conv2.stride = (1, 1)
            # self.net.layer3._modules['0'].downsample._modules['0'].stride = (1, 1)
            # self.net.layer4._modules['0'].conv2.stride = (1, 1)
            # self.net.layer4._modules['0'].downsample._modules['0'].stride = (1, 1)
            # self.output_size = 2048

        elif self.name == 'efficientnet-b0':
            self.net = EfficientNet.from_pretrained('efficientnet-b0')
            self.net._blocks._modules['5']._depthwise_conv.stride = [1, 1]
            self.net._blocks._modules['11']._depthwise_conv.stride = [1, 1]
            self.output_size = 1280

        self.bs = 32

    def _extract_features_for_batch(self, x):
        if self.name == 'resnet_18':
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)
            x = self.net.layer1(x)
            x = self.net.layer2(x)
            x = self.net.layer3(x)
            x = self.net.layer4(x)

        elif self.name == 'resnet_50':
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)
            x = self.net.layer1(x)
            x = self.net.layer2(x)  # notice that layer3 and 4 missing in this case
            if self.output_size == 2048:
                x = self.net.layer3(x)
                x = self.net.layer4(x)

        elif self.name == 'efficientnet-b0':
            x = self.net.extract_features(x)

        return x

    def forward(self, x):
        outputs = []
        # Extract feature from respective extractor
        for batch_idx in range(ceil(len(x) / self.bs)):
            outputs.extend(self._extract_features_for_batch(x[batch_idx * self.bs:(batch_idx + 1) * self.bs]))

        return torch.stack(outputs)


class SegmentationModel(nn.Module):
    def __init__(self, device):
        super(SegmentationModel, self).__init__()
        self.device = device
        self.feature_extractor = FeatureExtractor(name='efficientnet-b0')

        self.fast_conv1 = nn.Conv3d(
            in_channels=self.feature_extractor.output_size,  # 1280 for efficientnet, 512 for resnet
            out_channels=64,
            kernel_size=(16, 3, 3))  # TODO consider padding

        self.slow_conv1 = nn.Conv3d(
            in_channels=self.feature_extractor.output_size,
            out_channels=256,
            kernel_size=(4, 3, 3)  # TODO consider padding
            # TODO first version is without stride but with smaller kernel
        )
        # TODO: Unlike Track-RCNN, we can't have both of these convs set to identity from the start as identity kernel by definition requires in_channels==out_channels
        # TODO: But maybe we can still use that trick for the slow path

        mask_roi_pool = RoIAlign(output_size=14, spatial_scale=0.11, sampling_ratio=2)  # Can be checked again to make sure spatial_scale is correct

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(320, mask_layers, mask_dilation)

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                           mask_dim_reduced, num_classes=1)  # TODO either 1 or 2

        self.roi_head = RoIHeads(mask_roi_pool=mask_roi_pool, mask_head=mask_head, mask_predictor=mask_predictor)

        self.bs = 32
        # TODO lateral connection
        # TODO maskrcnn_loss supports discretinzation size (meaning they don't have to be the same size)

    def forward(self, x, bboxes, targets=None):
        image_features = self.feature_extractor(x)
        image_features = torch.cat([torch.zeros_like(image_features[:1, :, :, :].repeat(8, 1, 1, 1)), image_features,
                                    torch.zeros_like(image_features[:1, :, :, :].repeat(8, 1, 1, 1))])
        # 0 0 0 0 X X X X 0 0 0 0

        valid_features_mask = []

        for idx in range(8, len(image_features) - 8):
            if len(bboxes[idx - 8]) == 0:  # If no box predictions just skip
                valid_features_mask.append(0)
                continue
            else:
                valid_features_mask.append(1)

        # TODO modify this according to slowfast\
        # TODO instead of computing these, prepare them for batch creation at once
        total_loss = 0.
        pred_outputs = []
        for batch_idx in range(ceil(len(x) / self.bs)):
            feature_idxs = range(batch_idx * self.bs, min((batch_idx + 1) * self.bs, len(x)))
            slow_valid_features = []
            fast_valid_features = []
            batch_bboxes = []
            batch_targets = []
            for feature_idx in feature_idxs:
                if valid_features_mask[feature_idx] == 1:
                    image_feature_idx = feature_idx + 8
                    slow_valid_features.append(image_features[image_feature_idx - 2:image_feature_idx + 2].transpose(0, 1))
                    fast_valid_features.append(image_features[image_feature_idx - 8:image_feature_idx + 8].transpose(0, 1))

                    batch_bboxes.append(torch.cat(bboxes[feature_idx]).float().to(device=self.device))
                    batch_targets.append(torch.cat(targets[feature_idx]).float().to(device=self.device))

            if len(slow_valid_features) == 0:  # If no detections in batch, skip
                continue
            batch_slow_output_features = self.slow_conv1(torch.stack(slow_valid_features))
            batch_fast_output_features = self.fast_conv1(torch.stack(fast_valid_features))

            merged_features = torch.cat([batch_slow_output_features, batch_fast_output_features], dim=1)[:, :, 0, :, :]

            image_sizes = [tuple(x.shape[2:4])] * len(merged_features)
            if self.training:
                total_loss += self.roi_head(merged_features, batch_bboxes, image_sizes, batch_targets)
            else:
                if targets is not None:
                    loss, output = self.roi_head(merged_features, batch_bboxes, image_sizes, batch_targets)

                    total_loss += loss
                    pred_outputs.append(output)
                else:
                    output = self.roi_head(merged_features, batch_bboxes, image_sizes, batch_targets)
                    total_loss = -1
                    pred_outputs.append(output)

        if self.training:
            return total_loss
        else:
            return total_loss, torch.cat(pred_outputs)
