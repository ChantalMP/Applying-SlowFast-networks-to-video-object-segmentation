from math import ceil
import types
import torch
from torch import nn
# from roi_heads_custom import RoIHeads
# from torchvision.ops import RoIAlign
# from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
# from efficientnet_pytorch import EfficientNet
# from torchvision import models
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops


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
    def __init__(self, input_size):
        # TODO consider no padding
        super(SlowFastLayers, self).__init__()
        self.fast_conv1 = nn.Conv3d(
            in_channels=input_size,  # 1280 for efficientnet, 512 for resnet
            out_channels=32,
            kernel_size=(2, 3, 3),
            padding=(0, 1, 1))

        self.bn_f1 = nn.BatchNorm3d(32)

        self.slow_conv1 = nn.Conv3d(
            in_channels=input_size,
            out_channels=256,
            kernel_size=(2, 3, 3),
            padding=(0, 1, 1))
        # TODO maybe with stride

        self.bn_s1 = nn.BatchNorm3d(256)

        self.fast_conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=(0, 1, 1))

        self.bn_f2 = nn.BatchNorm3d(64)

        self.slow_conv2 = nn.Conv3d(
            in_channels=320,
            out_channels=512,
            kernel_size=(3, 3, 3),
            padding=(0, 1, 1)
        )

        self.bn_s2 = nn.BatchNorm3d(512)

        self.conv_f2s = nn.Conv3d(
            32,
            64,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )

        self.bn_f2s = nn.BatchNorm3d(64)

        self.relu = nn.ReLU(inplace=True)

    def fuse(self, slow, fast):
        fuse = self.conv_f2s(fast)
        fuse = self.bn_f2s(fuse)
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

        # Fuse
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


class SegmentationModel(nn.Module):
    def __init__(self, device, slow_pathway_size, fast_pathway_size):
        super(SegmentationModel, self).__init__()
        self.device = device
        self.maskrcnn_model = get_model_instance_segmentation(num_classes=2)
        self.maskrcnn_model.load_state_dict(torch.load('maskrcnn/maskrcnn_model_5.pth'))
        # When we use gt_masks, we want mask predictions for every box
        self.maskrcnn_model.roi_heads.postprocess_detections = types.MethodType(postprocess_detections, self.maskrcnn_model.roi_heads)
        # Freeze most of the weights
        for param in self.maskrcnn_model.backbone.parameters():
            param.requires_grad = False
        for param in self.maskrcnn_model.rpn.parameters():
            param.requires_grad = False

        # self.slow_pathway_size = slow_pathway_size
        # self.fast_pathway_size = fast_pathway_size
        #
        # self.slow_fast = SlowFastLayers(self.feature_extractor.output_size)

        self.bs = 16
        self.maskrcnn_bs = 8

    @torch.no_grad()
    def compute_maskrcnn_features(self, images_tensors):
        self.maskrcnn_model.eval()
        all_features = OrderedDict()
        # Extract feature from respective extractor
        for i in range(ceil(len(images_tensors) / self.maskrcnn_bs)):
            batch_imgs = images_tensors[i * self.maskrcnn_bs:(i + 1) * self.maskrcnn_bs].to(self.device)
            batch_features = self.maskrcnn_model.backbone(batch_imgs)
            for key, value in batch_features.items():
                if key not in all_features:
                    all_features[key] = value.cpu()
                else:
                    all_features[key] = torch.cat([all_features[key], value.cpu()])

        if self.training:
            self.maskrcnn_model.train()
        return all_features

    def _slice_features(self, features: OrderedDict, i):
        batch_features = OrderedDict()
        for key, value in features.items():
            batch_features[key] = value[i * self.bs:(i + 1) * self.bs].to(self.device)

        return batch_features

    def _slice_targets(self, targets, i):
        batch_targets = targets[i * self.bs:(i + 1) * self.bs]
        for batch_target in batch_targets:
            for key, value in batch_target.items():
                batch_target[key] = value.to(self.device)

        return batch_targets

    def forward(self, images, targets=None, padding=None):
        # overlap = self.fast_pathway_size // 2
        # padding is a tuple like (False,False) first one indicates need to append before the sequence, second one after the sequence
        valid_ids = []
        valid_targets = []
        valid_images = []
        for idx, target in enumerate(targets):
            if len(target) > 0:
                valid_targets.append(target)
                valid_images.append(images[idx])
                valid_ids.append(1)
            else:
                valid_ids.append(0)

        targets = valid_targets
        images = valid_images
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.maskrcnn_model.transform(images, targets)
        features = self.compute_maskrcnn_features(images.tensors)
        all_detector_losses = []
        all_detections = []

        for i in range(ceil(len(images.tensors) / self.bs)):
            batch_imgs = ImageList(images.tensors[i * self.bs:(i + 1) * self.bs].to(self.device), images.image_sizes[i * self.bs:(i + 1) * self.bs])
            batch_features = self._slice_features(features, i)
            batch_targets = self._slice_targets(targets, i)
            batch_original_image_sizes = original_image_sizes[i * self.bs:(i + 1) * self.bs]
            gt_proposals = [elem['boxes'] for elem in batch_targets]
            detections, detector_losses = self.maskrcnn_model.roi_heads(batch_features, gt_proposals, batch_imgs.image_sizes, batch_targets)
            detections = self.maskrcnn_model.transform.postprocess(detections, batch_imgs.image_sizes, batch_original_image_sizes)

            all_detector_losses.append(detector_losses)
            all_detections.extend(detections)

        losses = {}
        for loss_dict in all_detector_losses:
            for key, value in loss_dict.items():
                if key not in losses:
                    losses[key] = value
                else:
                    losses[key] += value

        # Append empty detection for non valid ids
        if not self.training:
            full_detections = []
            pointer = 0
            for valid_id in valid_ids:
                if valid_id:
                    full_detections.append(all_detections[pointer])
                    pointer += 1
                else:
                    full_detections.append({})

        return (losses, all_detections)
        # if padding is not None:
        #     if padding[0].item():
        #         image_features = torch.cat(
        #             [torch.zeros_like(image_features[:1, :, :, :].repeat(overlap, 1, 1, 1)), image_features])
        #
        #     else:  # As those bboxes correspond to imges only used for padding, we don't need them
        #         bboxes = bboxes[overlap:]
        #         if targets is not None:
        #             targets = targets[overlap:]
        #
        #     if padding[1].item():
        #         image_features = torch.cat(
        #             [image_features, torch.zeros_like(image_features[:1, :, :, :].repeat(overlap, 1, 1, 1))])
        #     else:
        #         bboxes = bboxes[:-overlap]
        #         if targets is not None:
        #             targets = targets[:-overlap]

        # valid_features_mask = []
        #
        # for idx in range(overlap, len(image_features) - overlap):
        #     if len(bboxes[idx - overlap]) == 0:  # If no box predictions just skip
        #         valid_features_mask.append(0)
        #         continue
        #     else:
        #         valid_features_mask.append(1)

        # total_loss = 0.
        # pred_outputs = []
        # for batch_idx in range(ceil(len(valid_features_mask) / self.bs)):
        #     feature_idxs = range(batch_idx * self.bs, min((batch_idx + 1) * self.bs, len(valid_features_mask)))
        #     slow_valid_features = []
        #     fast_valid_features = []
        #     batch_bboxes = []
        #     batch_targets = []
        #     for feature_idx in feature_idxs:
        #         if valid_features_mask[feature_idx] == 1:
        #             image_feature_idx = feature_idx + overlap
        #             # TODO right now slow sees the middle 4 frames, we should consider the option of seeing 4 frames through skipping
        #             slow_valid_features.append(image_features[
        #                                        image_feature_idx - self.slow_pathway_size // 2:image_feature_idx + self.slow_pathway_size // 2].transpose(
        #                 0, 1))
        #             fast_valid_features.append(image_features[
        #                                        image_feature_idx - self.fast_pathway_size // 2:image_feature_idx + self.fast_pathway_size // 2].transpose(
        #                 0, 1))
        #
        #             batch_bboxes.append(torch.cat(bboxes[feature_idx]).float().to(device=self.device))
        #             batch_targets.append(torch.cat(targets[feature_idx]).float().to(
        #                 device=self.device))  # TODO make runnable without targets
        #
        #     if len(slow_valid_features) == 0:  # If no detections in batch, skip
        #         continue
        #     batch_slow_output_features, batch_fast_output_features = self.slow_fast(torch.stack(slow_valid_features), torch.stack(fast_valid_features))
        #     # directly pass image features of middle frame as well # TODO Try this
        #     orig_resnet_features = torch.stack(slow_valid_features)[:, :,
        #                            self.slow_pathway_size // 2:self.slow_pathway_size // 2 + 1, :, :]
        #     merged_features = torch.cat([batch_slow_output_features, batch_fast_output_features, orig_resnet_features],  # orig_resnet_features
        #                                 dim=1)[:, :, 0, :, :]
        #
        #     image_sizes = [tuple(x.shape[2:4])] * len(merged_features)
        #     if self.training:
        #         total_loss += self.roi_head(merged_features, batch_bboxes, image_sizes, batch_targets)
        #     else:
        #         if targets is not None:
        #             loss, output = self.roi_head(merged_features, batch_bboxes, image_sizes, batch_targets)
        #
        #             total_loss += loss
        #             pred_outputs.append(output)
        #         else:
        #             output = self.roi_head(merged_features, batch_bboxes, image_sizes, batch_targets)
        #             total_loss = -1
        #             pred_outputs.append(output)
        #
        # if self.training:
        #     return total_loss
        # else:
        #     return total_loss, torch.cat(pred_outputs)
