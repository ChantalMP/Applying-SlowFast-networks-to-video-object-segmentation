import numpy as np
from PIL import Image
from matplotlib import patches
from matplotlib import pyplot as plt


def convert_mask_pred_to_ground_truth_format(img, box, mask, threshold=0.5):
    # resize mask
    resized_mask = Image.fromarray(mask)
    width, height = box[2] - box[0], box[3] - box[1]
    resized_mask = np.array(resized_mask.resize((width, height), resample=Image.ANTIALIAS))
    # Fill a image sizes mask with the values of resized mask at the corresponding location
    full_mask = np.zeros_like(img[:, :, 0])
    full_mask[box[1]:box[3], box[0]:box[2]] = resized_mask

    # threshold mask
    full_mask = (full_mask >= threshold).astype(np.float)

    return full_mask


def intersection_over_union(mask1: np.ndarray, mask2: np.ndarray):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)


def revert_normalization(img, means, stds):
    '''
    From pytorch: will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    So revert it by (input[channel]*std[channel]) + mean[channel]
    :param img:
    :return:
    '''
    reverted_img = np.zeros_like(img)
    for channel in range(reverted_img.shape[-1]):
        reverted_img[:, :, channel] = (img[:, :, channel] * stds[channel]) + means[channel]

    return reverted_img


def _visualize_image_with_properties(image: np.ndarray, masks=None, boxes=None):
    ax = plt.subplot(1, 1, 1)
    ax.set_axis_off()
    ax.imshow(image)
    ax.axis('off')
    if masks is not None:
        total_mask = np.zeros(image.shape[:2]).astype(np.bool)
        for mask in masks:
            total_mask = np.logical_or(total_mask, mask)

        full_mask = np.expand_dims(total_mask, axis=-1).repeat(4, axis=-1).astype(np.float)
        full_mask[:, :, 0] = 0.
        full_mask[:, :, 1] = 1
        full_mask[:, :, 2] = 0.

        ax.imshow(full_mask, alpha=0.3)

    elif boxes is not None:

        for box in zip(boxes):
            x = box[0][0].item()
            y = box[0][1].item()
            width = box[0][2].item() - x
            height = box[0][3].item() - y
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.show()


def visualize_image_with_properties(image: np.ndarray, masks=None, boxes=None, proposals=None):
    '''
    :param image: HxWxC
    :param masks: NxHxW
    :param boxes: Nx4
    :param proposals: Nx4
    '''
    if masks is not None:
        _visualize_image_with_properties(image, masks=masks)

    if boxes is not None:
        _visualize_image_with_properties(image, boxes=boxes)

    if proposals is not None:
        _visualize_image_with_properties(image, boxes=proposals)
