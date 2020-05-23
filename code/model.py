from math import ceil
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
        # 0 0 0 0 X X X X 0 0 0 0

        # resnet_features =
        # TODO this can actually be vectorized, but not sure if it is possible to do all at once do to memory constraints, shouldn't effect the outcome nonetheless
        valid_features_mask = []

        for idx in range(8, len(resnet_features) - 8):
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
            feature_idxs = range(batch_idx*self.bs, min((batch_idx+1)*self.bs, len(x)))
            slow_valid_features = []
            fast_valid_features = []
            batch_bboxes = []
            batch_targets = []
            for feature_idx in feature_idxs:
                if valid_features_mask[feature_idx] == 1:
                    resnet_feature_idx = feature_idx+8
                    slow_valid_features.append(resnet_features[resnet_feature_idx - 2:resnet_feature_idx + 2].transpose(0, 1))
                    fast_valid_features.append(resnet_features[resnet_feature_idx - 8:resnet_feature_idx + 8].transpose(0, 1))

                    batch_bboxes.append(torch.cat(bboxes[feature_idx]).float().to(device=self.device))
                    batch_targets.append(torch.cat(targets[feature_idx]).float().to(device=self.device))

            if len(slow_valid_features) == 0: # If no detections in batch, skip
                continue
            batch_slow_output_features = self.slow_conv1(torch.stack(slow_valid_features))
            batch_fast_output_features = self.fast_conv1(torch.stack(fast_valid_features))

            merged_features = torch.cat([batch_slow_output_features, batch_fast_output_features], dim=1)[:, :, 0, :, :]

            image_sizes = [tuple(x.shape[2:4])] * len(merged_features)
            if self.training:
                total_loss += self.roi_head(merged_features, batch_bboxes, image_sizes, batch_targets)
            else:
                pass

        if self.training:
            return total_loss
        else:
            full_output = []
            out_idx = 0
            for valid in valid_features_mask:
                if valid:
                    full_output.append(pred_outputs[out_idx])
                    out_idx += 1
                else:
                    full_output.append(torch.zeros_like(pred_outputs[0]))

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
