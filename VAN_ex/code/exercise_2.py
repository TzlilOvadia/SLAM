import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
import random
import utils.utils as utils
from models.Matcher import Matcher
from utils.plotters import gen_hist


def calc_y_coord_deviations(thresh=0):
    deviations = []
    for match in matchs:
        img1Idx, img2Idx = match[0].queryIdx, match[0].trainIdx
        _, y1 = kp1[img1Idx].pt
        _, y2 = kp2[img2Idx].pt
        delta = abs(y1 - y2)
        if delta > thresh:
            deviations.append(delta)
    return deviations


if __name__ == '__main__':
    img1, img2 = utils.read_images(6)
    sift = cv2.SIFT_create()
    matcher = Matcher(sift)
    matcher.detect_and_compute()
    matcher.find_matching_features(with_significance_test=False)
    matchs = matcher.get_matches()
    descriptors = matcher.get_dsc()
    kp1,kp2 = matcher.get_kp()
    deviations = calc_y_coord_deviations()

    gen_hist(data=deviations, bins=100)