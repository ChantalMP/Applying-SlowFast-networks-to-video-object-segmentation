import numpy as np
from PIL import Image


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
