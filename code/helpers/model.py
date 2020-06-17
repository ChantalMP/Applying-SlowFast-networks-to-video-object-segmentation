from math import ceil, floor
import types
import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
from helpers.constants import batch_size, maskrcnn_batch_size
from torchvision.models.detection.transform import resize_boxes


def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
    '''
    Overwrite the postprocessing from roiheads
    '''
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class SlowFastLayers(nn.Module):
    def __init__(self, input_size, device, slow_pathway_size, fast_pathway_size):
        super(SlowFastLayers, self).__init__()
        self.device = device
        self.slow_pathway_size = slow_pathway_size
        self.fast_pathway_size = fast_pathway_size

        kernel_size_slow1, kernel_size_slow2, kernel_size_slow3 = self._calc_kernel_sizes(self.slow_pathway_size)
        kernel_size_fast1, kernel_size_fast2, kernel_size_fast3 = self._calc_kernel_sizes(self.fast_pathway_size)

        kernel_size_f2s1, slow_out1, fast_out1 = self._calc_fuse_kernel_size(slow_in=self.slow_pathway_size,
                                                                             slow_kernel=kernel_size_slow1,
                                                                             fast_in=self.fast_pathway_size,
                                                                             fast_kernel=kernel_size_fast1)
        kernel_size_f2s2, _, _ = self._calc_fuse_kernel_size(slow_in=slow_out1, slow_kernel=kernel_size_slow2,
                                                             fast_in=fast_out1, fast_kernel=kernel_size_fast2)

        self.fast_conv1, self.bn_f1 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_fast1,
                                                             in_channels=input_size, out_channels=32)

        self.slow_conv1, self.bn_s1 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_slow1,
                                                             in_channels=input_size, out_channels=192)

        self.fast_conv2, self.bn_f2 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_fast2,
                                                             in_channels=32, out_channels=32)

        self.slow_conv2, self.bn_s2 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_slow2,
                                                             in_channels=256, out_channels=192)

        self.fast_conv3, self.bn_f3 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_fast3,
                                                             in_channels=32, out_channels=32)

        self.slow_conv3, self.bn_s3 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_slow3,
                                                             in_channels=256, out_channels=224)

        self.conv_f2s1, self.bn_f2s1 = self._init_fuse_and_bn(kernel_size_f2s1)

        self.conv_f2s2, self.bn_f2s2 = self._init_fuse_and_bn(kernel_size_f2s2)

        self.relu = nn.ReLU(inplace=True)

    def _init_conv_and_bn(self, temporal_kernelsize, in_channels, out_channels):
        conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(temporal_kernelsize, 3, 3),
            padding=(0, 1, 1))

        bn = nn.BatchNorm3d(out_channels)

        return conv, bn

    def _init_fuse_and_bn(self, temporal_kernelsize):
        conv_f2s = nn.Conv3d(
            32,
            64,
            kernel_size=[temporal_kernelsize, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )

        bn_f2s = nn.BatchNorm3d(64)

        return conv_f2s, bn_f2s

    def _calc_kernel_sizes(self, pathway_size):
        div = pathway_size // 3
        if pathway_size % 3 == 0:
            return (div, div + 1, div + 1)
        elif pathway_size % 3 == 1:
            return (div + 1, div + 1, div + 1)
        elif pathway_size % 3 == 2:
            return (div + 1, div + 1, div + 2)

    def _calc_fuse_kernel_size(self, slow_in, slow_kernel, fast_in, fast_kernel):
        out_slow = (slow_in - slow_kernel) + 1
        out_fast = (fast_in - fast_kernel) + 1
        fuse_kernel_size = out_fast - out_slow + 1
        return fuse_kernel_size, out_slow, out_fast

    def fuse(self, slow, fast, conv, bn):
        fuse = conv(fast)
        fuse = bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([slow, fuse], 1)
        return x_s_fuse, fast

    def forward(self, slow, fast):
        # First Conv Layer
        slow = self.slow_conv1(slow)
        slow = self.bn_s1(slow)
        slow = self.relu(slow)

        fast = self.fast_conv1(fast)
        fast = self.bn_f1(fast)
        fast = self.relu(fast)

        # Fuse1
        slow, fast = self.fuse(slow, fast, self.conv_f2s1, self.bn_f2s1)

        # Second Conv Layer
        slow = self.slow_conv2(slow)
        slow = self.bn_s2(slow)
        slow = self.relu(slow)

        fast = self.fast_conv2(fast)
        fast = self.bn_f2(fast)
        fast = self.relu(fast)

        # Fuse2
        slow, fast = self.fuse(slow, fast, self.conv_f2s2, self.bn_f2s2)

        # Second Conv Layer
        slow = self.slow_conv3(slow)
        slow = self.bn_s3(slow)

        fast = self.fast_conv3(fast)
        fast = self.bn_f3(fast)
        return slow, fast

    def temporally_enhance_features(self, slow_features, fast_features):
        # List of dicts to dict of lists
        slow_features = {k: [dic[k] for dic in slow_features] for k in slow_features[0]}
        fast_features = {k: [dic[k] for dic in fast_features] for k in fast_features[0]}
        merged_features = OrderedDict()
        for key in slow_features.keys():
            key_scale_slow_features = torch.stack(slow_features[key]).to(self.device).transpose(1, 2)
            key_scale_fast_features = torch.stack(fast_features[key]).to(self.device).transpose(1, 2)
            key_scale_slow_features, key_scale_fast_features = self.forward(key_scale_slow_features,
                                                                            key_scale_fast_features)

            merged_features[key] = torch.cat([key_scale_slow_features, key_scale_fast_features], dim=1).squeeze(dim=2)
            del key_scale_slow_features, key_scale_fast_features

        return merged_features


class SegmentationModel(nn.Module):
    def __init__(self, device, slow_pathway_size, fast_pathway_size):
        super(SegmentationModel, self).__init__()
        self.device = device
        self.maskrcnn_model = get_model_instance_segmentation(num_classes=2)
        self.maskrcnn_model.load_state_dict(torch.load('maskrcnn/maskrcnn_model.pth'))

        # Freeze most of the weights
        for param in self.maskrcnn_model.backbone.parameters():
            param.requires_grad = False
        for param in self.maskrcnn_model.rpn.parameters():
            param.requires_grad = False

        self.slow_pathway_size = slow_pathway_size
        self.fast_pathway_size = fast_pathway_size

        self.slow_fast = SlowFastLayers(256, device=device, slow_pathway_size=slow_pathway_size,
                                        fast_pathway_size=fast_pathway_size)

        self.bs = batch_size
        self.maskrcnn_bs = maskrcnn_batch_size
        self.maskrcnn_model.roi_heads.detections_per_img = 10

    @torch.no_grad()
    def compute_maskrcnn_features(self, images_tensors):
        self.maskrcnn_model.eval()
        all_features = OrderedDict()
        # Extract feature from respective extractor
        for i in range(ceil(len(images_tensors) / self.maskrcnn_bs)):
            batch_imgs = images_tensors[i * self.maskrcnn_bs:(i + 1) * self.maskrcnn_bs].to(self.device)
            batch_features = self.maskrcnn_model.backbone(batch_imgs)
            batch_features.pop('pool')  # remove unnecessary pool feature
            for key, value in batch_features.items():
                if key not in all_features:
                    all_features[key] = value.cpu()
                else:
                    all_features[key] = torch.cat([all_features[key], value.cpu()])

        if self.training:
            self.maskrcnn_model.train()
        return all_features

    def _slice_features(self, features: OrderedDict, image_feature_idx, pathway_size):
        batch_features = OrderedDict()
        for key, value in features.items():
            batch_features[key] = value[image_feature_idx - floor(pathway_size / 2):image_feature_idx + ceil(
                pathway_size / 2)]

        return batch_features

    def _targets_to_device(self, targets, device):
        for i in range(len(targets)):
            for key, value in targets[i].items():
                targets[i][key] = value.to(device)

    '''Padding at beginning and end of the sequence (we do not have real neighbouring feature maps there)'''

    def apply_padding(self, image_features):
        padding_count = self.fast_pathway_size // 2
        for key, image_feature in image_features.items():
            image_features[key] = torch.cat(
                [torch.zeros_like(image_feature[:1, :, :, :].repeat(padding_count, 1, 1, 1)), image_feature])

        for key, image_feature in image_features.items():
            image_features[key] = torch.cat(
                [image_feature, torch.zeros_like(image_feature[:1, :, :, :].repeat(padding_count, 1, 1, 1))])

        return image_features

    def forward(self, images, targets=None, optimizer=None):
        # TODO support targets is None (no skipping)
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        transformed_images, _ = self.maskrcnn_model.transform(images)
        image_features = self.compute_maskrcnn_features(transformed_images.tensors)

        '''Deal with imgs that have no objects in them'''
        valid_features_mask = []
        valid_targets = []
        valid_imgs = []
        for idx in range(len(image_features['0'])):
            if 'boxes' not in targets[idx] or len(targets[idx]['boxes']) == 0:  # If no box predictions just skip
                valid_features_mask.append(0)
                continue
            else:
                valid_features_mask.append(1)
                valid_targets.append(targets[idx])
                valid_imgs.append(images[idx])

        _, targets = self.maskrcnn_model.transform(valid_imgs, valid_targets)

        del valid_imgs, valid_targets

        image_features = self.apply_padding(image_features)
        images = transformed_images
        full_targets = []
        pointer = 0
        for valid_id in valid_features_mask:
            if valid_id:
                full_targets.append(targets[pointer])
                pointer += 1
            else:
                full_targets.append({})

        targets = full_targets

        total_loss = 0.
        all_detections = []

        for i in range(ceil(len(valid_features_mask) / self.bs)):
            feature_idxs = range(i * self.bs, min((i + 1) * self.bs, len(valid_features_mask)))
            slow_valid_features = []
            fast_valid_features = []
            batch_targets = []
            for feature_idx in feature_idxs:
                if valid_features_mask[feature_idx] == 1:
                    image_feature_idx = feature_idx + self.fast_pathway_size // 2
                    # right now slow sees the middle 4 frames, we should consider the option of seeing 4 frames through skipping
                    slow_valid_features.append(
                        self._slice_features(image_features, image_feature_idx, self.slow_pathway_size))
                    fast_valid_features.append(
                        self._slice_features(image_features, image_feature_idx, self.fast_pathway_size))

                    batch_targets.append(targets[feature_idx])  # TODO make runnable without targets

            if len(slow_valid_features) == 0:  # If no detections in batch, skip
                continue
            slow_fast_features = self.slow_fast.temporally_enhance_features(slow_valid_features, fast_valid_features)
            batch_original_image_sizes = original_image_sizes[i * self.bs:(i + 1) * self.bs]
            batch_image_sizes = images.image_sizes[0:1] * len(
                batch_original_image_sizes)  # Because all images in one sequence have the same size
            self._targets_to_device(batch_targets, self.device)
            proposals = [elem['proposals'] for elem in batch_targets]  # predicted boxes

            detections, detector_losses = self.maskrcnn_model.roi_heads(slow_fast_features, proposals,
                                                                        batch_image_sizes, batch_targets)
            detections = self.maskrcnn_model.transform.postprocess(detections, batch_image_sizes,
                                                                   batch_original_image_sizes)
            self._targets_to_device(detections, device=torch.device('cpu'))
            all_detections.extend(detections)

            del feature_idxs, slow_valid_features, fast_valid_features, batch_targets, slow_fast_features, proposals

            if self.training:
                losses = {}
                for key, value in detector_losses.items():
                    if key not in losses:
                        losses[key] = value
                    else:
                        losses[key] += value

                losses = sum(loss for loss in losses.values())
                total_loss += losses.item()
                losses.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Append empty detection for non valid ids
        if not self.training:
            full_detections = []
            pointer = 0
            for valid_id in valid_features_mask:
                if valid_id:
                    full_detections.append(all_detections[pointer])
                    pointer += 1
                else:
                    full_detections.append({})

        return (total_loss, all_detections)
