from collections import OrderedDict
from math import ceil

import torch

from helpers.model import SegmentationModel


class OsvosSegmentationModel(SegmentationModel):
    def __init__(self, device, slow_pathway_size, fast_pathway_size, batchsize=8):
        super(OsvosSegmentationModel, self).__init__(device, slow_pathway_size, fast_pathway_size)
        # Unfreeze all weights
        for param in self.maskrcnn_model.backbone.parameters():
            param.requires_grad = True
        for param in self.maskrcnn_model.rpn.parameters():
            param.requires_grad = True
        self.bs = batchsize
        self.maskrcnn_model.roi_heads.detections_per_img = 10
        self.accumulator = 0

    def compute_maskrcnn_features(self, images_tensors):
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

        return all_features

    #
    # def compute_rpn_proposals(self, images, features):
    #     all_proposals = []
    #     bs = 4
    #     for i in range(ceil(len(images.tensors) / bs)):
    #         batch_imgs = ImageList(images.tensors[i * bs:(i + 1) * bs].to(self.device),
    #                                images.image_sizes[i * bs:(i + 1) * bs])
    #         batch_features = self.batch_slice_features(features, begin=i * bs, end=(i + 1) * bs)
    #         batch_proposals, _ = self.maskrcnn_model.rpn(batch_imgs, batch_features, None)
    #         all_proposals.extend(batch_proposals)
    #
    #     return all_proposals

    def forward(self, images, target=None, optimizer=None):
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        transformed_images, _ = self.maskrcnn_model.transform(images)
        image_features = self.compute_maskrcnn_features(transformed_images.tensors)
        rpn_proposals = self.compute_rpn_proposals(transformed_images, image_features)
        image_feature_idx = self.fast_pathway_size // 2
        target['proposals'] = rpn_proposals[image_feature_idx].cpu()
        del rpn_proposals

        _, targets = self.maskrcnn_model.transform([images[0]], [target])
        images = transformed_images

        total_loss = 0.

        slow_valid_features = []
        fast_valid_features = []
        # right now slow sees the middle 4 frames, we should consider the option of seeing 4 frames through skipping
        slow_valid_features.append(self._slice_features(image_features, image_feature_idx, self.slow_pathway_size))
        fast_valid_features.append(self._slice_features(image_features, image_feature_idx, self.fast_pathway_size))
        slow_fast_features = self.slow_fast.temporally_enhance_features(slow_valid_features, fast_valid_features)
        self._targets_to_device(targets, self.device)
        proposals = [elem['proposals'] for elem in targets]  # predicted boxes
        detections, detector_losses = self.maskrcnn_model.roi_heads(slow_fast_features, proposals, images.image_sizes, targets)
        detections = self.maskrcnn_model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        self._targets_to_device(detections, device=torch.device('cpu'))
        del slow_valid_features, fast_valid_features, slow_fast_features, proposals

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
            self.accumulator += 1
            if self.accumulator == 4:
                optimizer.step()
                optimizer.zero_grad()
                self.accumulator = 0

        return (total_loss, detections)
