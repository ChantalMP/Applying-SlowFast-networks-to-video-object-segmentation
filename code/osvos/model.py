from math import ceil, floor
import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import OrderedDict
# from helpers.constants import batch_size, maskrcnn_batch_size
from helpers.model import SegmentationModel
from torchvision.models.detection.image_list import ImageList


class OsvosSegmentationModel(SegmentationModel):
    def __init__(self, device, slow_pathway_size, fast_pathway_size, batchsize=8):
        super(OsvosSegmentationModel, self).__init__(device, slow_pathway_size, fast_pathway_size)

        # Freeze most of the weights
        for param in self.maskrcnn_model.backbone.parameters():
            param.requires_grad = True
        for param in self.maskrcnn_model.rpn.parameters():
            param.requires_grad = True
        self.bs = batchsize
        self.maskrcnn_model.roi_heads.detections_per_img = 10

    def batch_slice_features(self, features: OrderedDict, begin, end):
        batch_features = OrderedDict()
        for key, value in features.items():
            batch_features[key] = value[begin:end].to(self.device)

        return batch_features

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

    def forward(self, images, targets=None, optimizer=None):
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        transformed_images, _ = self.maskrcnn_model.transform(images)
        image_features = self.compute_maskrcnn_features(transformed_images.tensors)
        rpn_proposals = self.compute_rpn_proposals(transformed_images, image_features)
        for elem, proposal in zip(targets, rpn_proposals):
            elem['proposals'] = proposal.cpu()
        del rpn_proposals

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
