import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

MAC_OS_PATH = "dataset/sequences/05/"
WINDOWS_OS_PATH = "dataset\\sequences\\05\\"
SEP = "\\" if os.name == 'nt' else "/"
DATA_PATH = WINDOWS_OS_PATH if os.name == 'nt' else MAC_OS_PATH
# noinspection PyUnresolvedReferences


class Matcher:
    """A class for matching features between two images using a specified algorithm and matcher."""
    def __init__(self, algo, matcher=cv2.BFMatcher(), file_index=0, threshold=.2):
        """
        Initialize a Matcher object with an algorithm, a matcher, and a file index.
        :param algo: A cv2 algorithm object (e.g. cv2.SIFT_create(), cv2.AKAZE_create(), cv2.ORB_create()).
        :param matcher: A cv2 matcher object (default is cv2.BFMatcher())
        :param file_index: An integer file index for the images to match.
        :param threshold: A float in the range on [0,1], used in order to filter descriptors by their significance.
        """
        self.algo = algo
        self.matcher = matcher
        self.threshold = threshold
        self._img1, self._img2 = None, None
        self.read_images(idx=file_index)
        self._img1_kp, self._img1_dsc = None, None
        self._img2_kp, self._img2_dsc = None, None

    def read_images(self, idx) -> (np.ndarray, np.ndarray):
        """
        Read two images (img1 and img2) given an index and assign them to the Matcher object.
        :param idx: An integer file index for the images to match.
        :return: A tuple of the two images (np.ndarray) assigned to the Matcher object.
        """
        img_name = '{:06d}.png'.format(idx)
        self._img1 = cv2.imread(DATA_PATH + f'image_0{SEP}' + img_name, 0)
        self._img2 = cv2.imread(DATA_PATH + f'image_1{SEP}' + img_name, 0)

    def detect_and_compute(self):
        """Detect and compute the key-points and descriptors for the img1 and img2 images using the algorithm specified
        in the constructor."""
        self._img1_kp, self._img1_dsc = self.algo.detectAndCompute(self._img1, None)
        self._img2_kp, self._img2_dsc = self.algo.detectAndCompute(self._img2, None)

    def find_matching_features(self):
        """Find matching features between the img1 and img2 images using the matcher specified in the constructor.
         Apply a threshold to the matches to filter out poor matches and then call the drawMatchesKnn() function to
         display the matching features in a new image."""
        # BFMatcher with default params
        matches = self.matcher.knnMatch(self._img1_dsc, self._img2_dsc, k=2)
        filtered = self.apply_threshold(matches)
        img3 = cv2.drawMatchesKnn(self._img1, self._img1_kp, self._img2, self._img2_kp, filtered, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3)
        plt.show()

    def apply_threshold(self, matches: tuple):
        """Filter matches based on a threshold value and return the filtered matches.
        :param matches: A tuple of matches to filter.
        :param thresh: A float threshold value to filter matches (default is 0.1).
        :return:  A list of filtered matches.
        """
        filtered = []
        for m, n in matches:
            if m.distance < self.threshold * n.distance:
                filtered.append([m])
        return filtered


if __name__ == '__main__':
    # Initialize SIFT, AKAZE, and ORB algorithms
    sift = cv2.SIFT_create()
    akaze = cv2.AKAZE_create()
    orb = cv2.ORB_create()

    # Create matchers for SIFT, AKAZE, and ORB algorithms
    sift_matcher = Matcher(sift)
    akaze_matcher = Matcher(akaze)
    orb_matcher = Matcher(orb)

    # Detect and compute keypoints and descriptors for all matchers
    sift_matcher.detect_and_compute()
    akaze_matcher.detect_and_compute()
    orb_matcher.detect_and_compute()

    # Find matching features and display results for all matchers
    sift_matcher.find_matching_features()
    akaze_matcher.find_matching_features()
    orb_matcher.find_matching_features()