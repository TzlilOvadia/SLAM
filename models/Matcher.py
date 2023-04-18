import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from utils.utils import read_images

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
        self._matches = None

    def read_images(self, idx) -> (np.ndarray, np.ndarray):
        """
        Read two images (img1 and img2) given an index and assign them to the Matcher object.
        :param idx: An integer file index for the images to match.
        :return: None
        """
        self._img1, self._img2 = read_images(idx)

    def get_matches(self)->np.ndarray:
        return self._matches

    def get_kp(self):
        return self._img1_kp, self._img2_kp

    def get_dsc(self):
        return self._img1_dsc,self._img2_dsc


    def detect(self, debug=False):
        """Detect key-points for the img1 and img2 images using the algorithm specified
        in the constructor."""
        self._img1_kp = self.algo.detect(self._img1, None)
        self._img2_kp = self.algo.detect(self._img2, None)
        if debug:
            print(f"number of kp in im1: {len(self._img1_kp)}, and in im2: {len(self._img2_kp)}")
            im1 = cv2.drawKeypoints(self._img1, self._img1_kp, None)
            im2 = cv2.drawKeypoints(self._img2, self._img2_kp, None)
            plt.imshow(im1)
            plt.show()
            plt.imshow(im2)
            plt.show()

    def compute(self, debug=False):
        """Compute the descriptors for the key points found using the algorithm specified
        in the constructor."""
        self._img1_dsc = self.algo.compute(self._img1, self._img1_kp, None)[1]
        self._img2_dsc = self.algo.compute(self._img2, self._img2_kp, None)[1]
        if debug:
            print(f"first descriptor in im1: \n{self._img1_dsc[0]}")
            print(f"first descriptor in im2: \n{self._img2_dsc[0]}")

    def detect_and_compute(self):
        """Detect and compute the key-points and descriptors for the img1 and img2 images using the algorithm specified
        in the constructor."""
        self._img1_kp, self._img1_dsc = self.algo.detectAndCompute(self._img1, None)
        self._img2_kp, self._img2_dsc = self.algo.detectAndCompute(self._img2, None)

    def find_matching_features(self, with_significance_test=True, debug=False):
        """Find matching features between the img1 and img2 images using the matcher specified in the constructor.
         Apply a threshold to the matches to filter out poor matches and then call the drawMatchesKnn() function to
         display the matching features in a new image."""
        # BFMatcher with default params
        if with_significance_test:
            self._matches = self.matcher.knnMatch(self._img1_dsc, self._img2_dsc, k=2)
            self.apply_threshold(debug)
        else:
            self._matches = self.matcher.knnMatch(self._img1_dsc, self._img2_dsc, k=1)

    def apply_threshold(self, debug=False):
        """Filter matches based on a threshold value and return the filtered matches.
        :param thresh: A float threshold value to filter matches (default is 0.1).
        :return:  None
        """
        filtered = []
        for m, n in self._matches:
            if m.distance < self.threshold * n.distance:
                filtered.append([m])
        if debug:
            print(f"When applying significance test with threshold {self.threshold}, out of {len(self._matches)} matches {len(filtered)} matches remained")
        self._matches = filtered

    def draw_matches(self, num_of_matches=5000, debug=False):
        """
        This function draws the matches found by the matching algorithm between the two input images.
        :param num_of_matches: Maximum number of matches to be randomly chosen and displayed
        :return: None
        """
        matches = self._matches
        im1, im2 = self._img1, self._img2
        current_num_of_matches = len(self._matches)
        if current_num_of_matches >= num_of_matches:
           matches = random.sample(matches, num_of_matches)
        print(f"drawing {len(matches)} matches")
        if debug:
            im_1_pt = self._img1_kp[212]
            im_2_pt = self._img2_kp[135]
            im1 = cv2.circle(self._img1,(np.int(im_1_pt.pt[0]),np.int(im_1_pt.pt[1])),10, 3,thickness=3, lineType=8, shift=0)
            im2 = cv2.circle(self._img2, (np.int(im_2_pt.pt[0]), np.int(im_2_pt.pt[1])), 10, 3, thickness=3, lineType=8,
                             shift=0)
        img3 = cv2.drawMatchesKnn(im1, self._img1_kp, im2, self._img2_kp, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(15, 15))
        plt.imshow(img3)
        plt.show()