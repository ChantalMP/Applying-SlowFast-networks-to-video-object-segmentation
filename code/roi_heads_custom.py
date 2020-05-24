import torch

import torch.nn.functional as F
from torch import Tensor

from torchvision.ops import boxes as box_ops

from torchvision.ops import roi_align
from torchvision.ops._utils import convert_boxes_to_roi_format


def maskrcnn_inference(x, labels):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    Arguments:
        x (Tensor): the mask logits
        labels (list[BoxList]): bounding boxes that are used as
            reference, one for ech image

    Returns:
        results (list[BoxList]): one BoxList for each image, containing
            the extra field mask
    """
    mask_prob = x.sigmoid()

    # select masks coresponding to the predicted classes
    num_masks = x.shape[0]
    boxes_per_image = [len(l) for l in labels]
    labels = torch.cat(labels)
    index = torch.arange(num_masks, device=labels.device)
    mask_prob = mask_prob[index, labels][:, None]

    if len(boxes_per_image) == 1:
        # TODO : remove when dynamic split supported in ONNX
        # and remove assignment to mask_prob_list, just assign to mask_prob
        mask_prob_list = [mask_prob]
    else:
        mask_prob_list = mask_prob.split(boxes_per_image, dim=0)

    return mask_prob_list


def project_masks_on_boxes(gt_masks, boxes, M):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    rois = torch.cat([torch.zeros_like(boxes)[:, :1], boxes],
                     dim=1)  # TODO not sure at all about this, but we have to append something that corresponds to batch idx, see convert_boxes_to_roi_format
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1.)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    mask_targets = [
        project_masks_on_boxes(m, p, discretization_size)
        for m, p in zip(gt_masks, proposals)
    ]
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    from matplotlib import pyplot as plt
    plt.imshow(mask_targets[0].cpu().numpy(),cmap='Greys')
    plt.show()
    plt.imshow(mask_logits[0].sigmoid().detach().cpu().numpy(),cmap='Greys')
    plt.show()
    # TODO make sure to use correct prososal_mask to correct gt_mask
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_targets)
    return mask_loss


class RoIHeads(torch.nn.Module):

    def __init__(self,
                 mask_roi_pool,
                 mask_head,
                 mask_predictor,
                 ):
        super(RoIHeads, self).__init__()

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
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

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self, features, mask_proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        # TODO convert maskproposal to correct roi format
        mask_features = self.mask_roi_pool(features, mask_proposals)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)[:, 0, :, :]

        if self.training:
            assert targets is not None
            assert mask_logits is not None

            gt_masks = [t for t in targets]
            rcnn_loss_mask = maskrcnn_loss(
                mask_logits, mask_proposals,
                gt_masks)
            return rcnn_loss_mask
        else:
            return mask_logits.sigmoid()
