from math import ceil, floor

import torch

from helpers.model import SegmentationModel


class OsvosSegmentationModel(SegmentationModel):
    def __init__(self, device, slow_pathway_size, fast_pathway_size, batchsize=8, cfg=None):
        super(OsvosSegmentationModel, self).__init__(device, slow_pathway_size, fast_pathway_size)
        if cfg is None:
            # Unfreeze all weights
            for param in self.maskrcnn_model.backbone.parameters():
                param.requires_grad = True
            for param in self.maskrcnn_model.rpn.parameters():
                param.requires_grad = True
        else:
            self.requires_grad = {'backbone': True, 'slowfast': True}
            if cfg.freeze == 'SF':
                self.requires_grad['slowfast'] = False
            elif cfg.freeze == 'BB_SF':
                self.requires_grad['slowfast'] = False
                self.requires_grad['backbone'] = False

            for param in self.maskrcnn_model.backbone.parameters():
                param.requires_grad = self.requires_grad['backbone']
            for param in self.maskrcnn_model.rpn.parameters():
                param.requires_grad = self.requires_grad['backbone']
            for param in self.slow_fast.parameters():
                param.requires_grad = self.requires_grad['slowfast']

        self.bs = batchsize
        self.maskrcnn_model.roi_heads.detections_per_img = 10
        self.accumulator = 0

    def forward(self, images, target=None, optimizer=None):
        self.features_cache = {}
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        transformed_images, _ = self.maskrcnn_model.transform(images)

        _, targets = self.maskrcnn_model.transform([images[0]], [target])
        images = transformed_images

        total_loss = 0.
        padded_idx = self.fast_pathway_size // 2
        indices = range(padded_idx - floor(self.fast_pathway_size / 2), padded_idx + ceil(self.fast_pathway_size / 2))
        with torch.set_grad_enabled(self.requires_grad['backbone']):
            image_features = self.compute_maskrcnn_features(transformed_images.tensors, indices)
        sliced_features = self._index_features(image_features, padded_idx, padded_idx + 1)
        target = self._targets_to_device(targets, self.device)
        with torch.set_grad_enabled(self.requires_grad['backbone']):
            rpn_proposals, proposal_loses = self.compute_rpn_proposals(transformed_images.tensors[padded_idx:padded_idx + 1],
                                                                       transformed_images.image_sizes[padded_idx:padded_idx + 1],
                                                                       sliced_features,
                                                                       target)
        target[0]['proposals'] = rpn_proposals[0]
        slow_valid_features = [
            self._slice_features(image_features, image_feature_idx=padded_idx, pathway_size=self.slow_pathway_size)]
        fast_valid_features = [image_features]

        with torch.set_grad_enabled(self.requires_grad['backbone'] or self.requires_grad['slowfast']):
            slow_fast_features = self.slow_fast.temporally_enhance_features(slow_valid_features, fast_valid_features)
        proposals = [elem['proposals'] for elem in target]  # predicted boxes
        detections, detector_losses = self.maskrcnn_model.roi_heads(slow_fast_features, proposals, images.image_sizes, target)
        detections = self.maskrcnn_model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        detections = self._targets_to_device(detections, device=torch.device('cpu'))
        del slow_valid_features, fast_valid_features, slow_fast_features, proposals, target

        if self.training:
            losses = {}
            for key, value in detector_losses.items():
                if key not in losses:
                    losses[key] = value
                else:
                    losses[key] += value

            for key, value in proposal_loses.items():
                if key not in losses:
                    losses[key] = value
                else:
                    losses[key] += value

            losses = sum(loss for loss in losses.values())
            total_loss += losses.item()
            losses.backward()
            self.accumulator += 1
            if self.accumulator == 2:
                optimizer.step()
                optimizer.zero_grad()
                self.accumulator = 0

        return (total_loss, detections)
