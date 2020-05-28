from math import ceil
import torch
from torch import nn
from roi_heads_custom import RoIHeads
from torchvision.ops import RoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from efficientnet_pytorch import EfficientNet
from torchvision import models

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

class SlowFastLayers(nn.Module):
    def __init__(self, input_size):
        # TODO consider no padding
        super(SlowFastLayers, self).__init__()
        self.fast_conv1 = nn.Conv3d(
            in_channels=input_size,  # 1280 for efficientnet, 512 for resnet
            out_channels=64,
            kernel_size=(8, 3, 3),
            padding=(0, 1, 1))

        self.bn_f1 = nn.BatchNorm3d(64)

        self.slow_conv1 = nn.Conv3d(
            in_channels=input_size,
            out_channels=512,
            kernel_size=(2, 3, 3),
            padding=(0, 1, 1))
        # TODO maybe with stride

        self.bn_s1 = nn.BatchNorm3d(512)

        self.fast_conv2 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(9, 3, 3),
            padding=(0, 1, 1))

        self.bn_f2 = nn.BatchNorm3d(128)

        self.slow_conv2 = nn.Conv3d(
            in_channels=640,
            out_channels=896,
            kernel_size=(3, 3, 3),
            padding=(0, 1, 1)
        )

        self.bn_s2 = nn.BatchNorm3d(896)

        self.conv_f2s = nn.Conv3d(
            64,
            128,
            kernel_size=[3, 1, 1],
            stride=[3, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )

        self.bn_f2s = nn.BatchNorm3d(128)

        self.relu = nn.ReLU(inplace=True)

    def fuse(self, slow, fast):
        fuse = self.conv_f2s(fast)
        fuse = self.bn_f2s(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([slow, fuse], 1)
        return x_s_fuse, fast

    def forward(self, slow, fast):
        #First Conv Layer
        slow = self.slow_conv1(slow)
        slow = self.bn_s1(slow)
        slow = self.relu(slow)

        fast = self.fast_conv1(fast)
        fast = self.bn_f1(fast)
        fast = self.relu(fast)

        #Fuse
        slow, fast = self.fuse(slow, fast)

        # Second Conv Layer
        slow = self.slow_conv2(slow)
        slow = self.bn_s2(slow)
        slow = self.relu(slow)

        fast = self.fast_conv2(fast)
        fast = self.bn_f2(fast)
        fast = self.relu(fast)
        # TODO maybe don't use Relu at the end, but SlowFast seems to do it
        return slow, fast

class SegmentationModel(nn.Module):
    def __init__(self, device, slow_pathway_size, fast_pathway_size):
        super(SegmentationModel, self).__init__()
        self.device = device
        self.feature_extractor = FeatureExtractor(name='resnet_18')
        self.slow_pathway_size = slow_pathway_size
        self.fast_pathway_size = fast_pathway_size

        self.slow_fast = SlowFastLayers(self.feature_extractor.output_size)

        mask_roi_pool = RoIAlign(output_size=14, spatial_scale=0.11, sampling_ratio=2)  # Can be checked again to make sure spatial_scale is correct

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(1536, mask_layers, mask_dilation)

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                           mask_dim_reduced, num_classes=1)  # TODO either 1 or 2

        self.roi_head = RoIHeads(mask_roi_pool=mask_roi_pool, mask_head=mask_head, mask_predictor=mask_predictor,
                                 device=device)

        self.bs = 16

    def forward(self, x, bboxes, targets=None, padding=None):
        overlap = self.fast_pathway_size // 2
        # padding is a tuple like (False,False) first one indicates need to append before the sequence, second one after the sequence
        image_features = self.feature_extractor(x)
        if padding is not None:
            if padding[0].item():
                image_features = torch.cat(
                    [torch.zeros_like(image_features[:1, :, :, :].repeat(overlap, 1, 1, 1)), image_features])

            else:  # As those bboxes correspond to imges only used for padding, we don't need them
                bboxes = bboxes[overlap:]
                if targets is not None:
                    targets = targets[overlap:]

            if padding[1].item():
                image_features = torch.cat(
                    [image_features, torch.zeros_like(image_features[:1, :, :, :].repeat(overlap, 1, 1, 1))])
            else:
                bboxes = bboxes[:-overlap]
                if targets is not None:
                    targets = targets[:-overlap]

        valid_features_mask = []

        for idx in range(overlap, len(image_features) - overlap):
            if len(bboxes[idx - overlap]) == 0:  # If no box predictions just skip
                valid_features_mask.append(0)
                continue
            else:
                valid_features_mask.append(1)

        total_loss = 0.
        pred_outputs = []
        for batch_idx in range(ceil(len(valid_features_mask) / self.bs)):
            feature_idxs = range(batch_idx * self.bs, min((batch_idx + 1) * self.bs, len(valid_features_mask)))
            slow_valid_features = []
            fast_valid_features = []
            batch_bboxes = []
            batch_targets = []
            for feature_idx in feature_idxs:
                if valid_features_mask[feature_idx] == 1:
                    image_feature_idx = feature_idx + overlap
                    # TODO right now slow sees the middle 4 frames, we should consider the option of seeing 4 frames through skipping
                    slow_valid_features.append(image_features[
                                               image_feature_idx - self.slow_pathway_size // 2:image_feature_idx + self.slow_pathway_size // 2].transpose(
                        0, 1))
                    fast_valid_features.append(image_features[
                                               image_feature_idx - self.fast_pathway_size // 2:image_feature_idx + self.fast_pathway_size // 2].transpose(
                        0, 1))

                    batch_bboxes.append(torch.cat(bboxes[feature_idx]).float().to(device=self.device))
                    batch_targets.append(torch.cat(targets[feature_idx]).float().to(
                        device=self.device))  # TODO make runnable without targets

            if len(slow_valid_features) == 0:  # If no detections in batch, skip
                continue
            batch_slow_output_features, batch_fast_output_features = self.slow_fast(torch.stack(slow_valid_features), torch.stack(fast_valid_features))
            # directly pass image features of middle frame as well
            orig_resnet_features = torch.stack(slow_valid_features)[:, :,
                                   self.slow_pathway_size // 2:self.slow_pathway_size // 2 + 1, :, :]
            merged_features = torch.cat([batch_slow_output_features, batch_fast_output_features, orig_resnet_features],
                                        dim=1)[:, :, 0, :, :]

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
