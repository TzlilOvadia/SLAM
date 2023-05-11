import os
import cv2
import numpy as np

MAC_OS_PATH = "../dataset/sequences/05/"
WINDOWS_OS_PATH = "../dataset/sequences/05\\"
SEP = "\\" if os.name == 'nt' else "/"
DATA_PATH = WINDOWS_OS_PATH if os.name == 'nt' else MAC_OS_PATH


def read_images(idx) -> (np.ndarray, np.ndarray):
    """
    Read two images (img1 and img2) given an index and assign them to the Matcher object.
    :param idx: An integer file index for the images to match.
    :return: None
    """
    img_name = '{:06d}.png'.format(idx)
    _img1 = cv2.imread(DATA_PATH + f'image_0{SEP}' + img_name, 0)
    _img2 = cv2.imread(DATA_PATH + f'image_1{SEP}' + img_name, 0)

    return _img1, _img2


def coords_from_kps(matches, kp1, kp2):
    """
    Extracts the (x, y) coordinates of matched features in two images and their corresponding indices in the original
    keypoint arrays.

    Parameters:
    -----------
    matches : list of cv2.DMatch objects
        A list of matching feature descriptors, each containing the queryIdx and trainIdx attributes.
    kp1 : list of cv2.KeyPoint objects
        A list of keypoint objects representing features in the first image.
    kp2 : list of cv2.KeyPoint objects
        A list of keypoint objects representing features in the second image.

    Returns:
    --------
    x1, y1, x2, y2 : ndarray
        Four 1D arrays of shape (num_matches,) containing the x and y coordinates of matched features in the first
        and second images, respectively.
    indices_mapping : ndarray
        An array of shape (2, num_matches) containing the corresponding indices of the matched features in the
        original kp1 and kp2 arrays.
    """
    query_idxs = np.array([match[0].queryIdx for match in matches])
    train_idxs = np.array([match[0].trainIdx for match in matches])
    indices_mapping = np.stack((query_idxs,train_idxs))

    x1, y1 = np.array([kp1[idx].pt for idx in query_idxs]).T
    x2, y2 = np.array([kp2[idx].pt for idx in train_idxs]).T

    return x1, y1, x2, y2, indices_mapping


def read_cameras():
    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:] # skip first token
        l2 = f.readline().split()[1:] # skip first token
        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)
        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)
        k = m1[:, :3]
        mat1 = np.linalg.inv(k) @ m1
        mat2 = np.linalg.inv(k) @ m2
        return k, mat1, mat2


def _rectified_stereo_classifier(y1: np.ndarray, y2:np.ndarray, thresh=1):
    """
    An helper function for rectificatied_stereo_pattern.
    :param y1: The y-coordinate of all kp1.pt
    :param y2: The y-coordinate of the corresponding y1 (matched by taking the trainIdx)
    :param thresh: Threshold value to determine how to reject matching points
    :return: inliers and outliers nd arrays
    """
    deviations = np.abs(y1 - y2)
    inliers_idx = np.argwhere(deviations <= thresh)[:,0]
    outliers_idx = np.argwhere(deviations > thresh)[:,0]
    return inliers_idx, outliers_idx


def rectificatied_stereo_pattern(y1, y2, indices_mapping, thresh=1):
    """
    For each pair of points, this function will return a classification of points to two groups: Inliers and Outliers.
    This is done for each image.
    :param y1: The y-coordinate of all kp1.pt (Left image of the KITTI dataset)
    :param y2: The y-coordinate of the corresponding y1, matched by taking the trainIdx (Right image of the KITTI dataset).
    :param thresh: Threshold value to determine how to reject matching points
    :return:
    """
    inliers_idx, outliers_idx = _rectified_stereo_classifier(y1, y2, thresh)
    img1in, img2in = indices_mapping[0, inliers_idx], indices_mapping[1, inliers_idx]
    img1out, img2out = indices_mapping[0, outliers_idx], indices_mapping[1, outliers_idx]
    return img1in, img2in, img1out, img2out

def array_to_dict(arr):
    """
    Convert a 2D numpy array to a dictionary.

    Parameters:
    -----------
    arr : ndarray
        A 2D numpy array with shape (2, N).

    Returns:
    --------
    dict
        A dictionary where all values in the 1st row are keys, and all values in the 2nd row are values.
    """
    dict_obj = {}
    for i in range(arr.shape[1]):
        key = arr[0, i]
        value = arr[1, i]
        dict_obj[key] = value
    return dict_obj