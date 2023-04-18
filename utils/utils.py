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


def coords_from_kps(matches, kp1,kp2):
    query_idxs = np.array([match[0].queryIdx for match in matches])
    train_idxs = np.array([match[0].trainIdx for match in matches])
    indices_mapping = np.stack((query_idxs,train_idxs))
    x1, y1 = np.array([kp1[idx].pt for idx in query_idxs]).T
    x2, y2 = np.array([kp2[idx].pt for idx in train_idxs]).T

    return x1, y1, x2, y2, indices_mapping

