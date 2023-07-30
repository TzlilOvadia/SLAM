import cv2
import numpy as np
from matplotlib import pyplot as plt

from models.Constants import *
from models.TrackDatabase import TrackDatabase

from utils.utils import read_images
from utils.plotters import draw_matches
# noinspection PyUnresolvedReferences
FILTERED_MATCHES = "FILTERED_MATCHES"


class Matcher:
    """A class for matching features between two images using a specified algorithm and matcher."""
    def __init__(self, algo=cv2.SIFT_create(), matcher=cv2.BFMatcher(), file_index=0, threshold=.6, display=HORIZONTAL_REPRESENTATION):
        """
        Initialize a Matcher object with an algorithm, a matcher, and a file index.
        :param algo: A cv2 algorithm object (e.g. cv2.SIFT_create(), cv2.AKAZE_create(), cv2.ORB_create()).
        :param matcher: A cv2 matcher object (default is cv2.BFMatcher())
        :param file_index: An integer file index for the images to match.
        :param threshold: A float in the range on [0,1], used in order to filter descriptors by their significance.
        """
        # Added a caching option which saves everything related to a particular file index (key)

        self.cache = {}
        self.display = display
        self._file_index = file_index
        self.algo = algo
        self.matcher = matcher
        self.threshold = threshold
        self._img1, self._img2 = None, None
        self.read_images(idx=file_index)
        self._filtered_matches = None

        self._img1_kp, self._img1_dsc = None, None
        self._img2_kp, self._img2_dsc = None, None
        self._matches = None

    def read_images(self, idx) -> (np.ndarray, np.ndarray):
        """
        Read two images (img1 and img2) given an index and assign them to the Matcher object.
        :param idx: An integer file index for the images to match.
        :return: None
        """
        self._file_index = idx
        self._img1, self._img2 = read_images(idx)

        # I Changed this function to keep each frame array into the cache for later use
        if idx not in self.cache:
            self.cache[idx] = {FRAMES: (self._img1, self._img2), LEFT: None, RIGHT: None}
        else:
            self.cache[idx][FRAMES] = (self._img1, self._img2)

    def get_matches(self, idx=None) -> np.ndarray:
        if idx is None:
            return self._matches
        return self.cache[idx][MATCHES]

    def get_filtered_matches(self, idx=None) -> np.ndarray:
        if idx is None:
            return self._filtered_matches
        return self.cache[idx][FILTERED_MATCHES]

    def set_matches(self, matches, idx=None):
        self.cache[idx][MATCHES] = matches

    def get_kp(self, idx=None):
        if idx is None:
            return self._img1_kp, self._img2_kp
        return self.cache[idx][LEFT][0], self.cache[idx][RIGHT][0]

    def get_feature_location_frame(self, frameId: int, kp: int, loc: int):
        kpi = self.get_kp(frameId)[loc]
        x, y = kpi[kp].pt
        return x, y

    def get_dsc(self):
        return self._img1_dsc, self._img2_dsc

    def get_images(self, file_index=0):
        if file_index in self.cache:
            return self.cache[file_index][FRAMES]
        else:
            self.read_images(file_index)
            return self._img1, self._img2

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

    def detect_and_compute(self, file_index=0):
        """Detect and compute the key-points and descriptors for the img1 and img2 images using the algorithm specified
        in the constructor."""
        if self._file_index != file_index:
            self.read_images(file_index)
        self._img1_kp, self._img1_dsc = self.algo.detectAndCompute(self._img1, None)
        self._img2_kp, self._img2_dsc = self.algo.detectAndCompute(self._img2, None)
        self.cache[self._file_index][LEFT] = (np.copy(self._img1_kp), np.copy(self._img1_dsc))
        self.cache[self._file_index][RIGHT] = (np.copy(self._img2_kp), np.copy(self._img2_dsc))

    def find_matching_features(self, with_significance_test=False, debug=False):
        """Find matching features between the img1 and img2 images using the matcher specified in the constructor.
         Apply a threshold to the matches to filter out poor matches and then call the drawMatchesKnn() function to
         display the matching features in a new image."""
        # BFMatcher with default params
        if with_significance_test:
            matches = self.matcher.knnMatch(self._img1_dsc, self._img2_dsc, k=2)
            self._matches = matches
            self.apply_threshold(debug)
            # Retrieve keypoints and descriptors for filtered matches
            # self._img1_kp = [self._img1_kp[m[0].queryIdx] for m in self._matches]
            # self._img2_kp = [self._img2_kp[m[0].trainIdx] for m in self._matches]
            # self.cache[self._file_index][LEFT] = self._img1_kp
            # self.cache[self._file_index][RIGHT]=self._img2_kp
            # descriptors1 = np.array([self._img1_dsc[m[0].queryIdx] for m in self._matches])
            # descriptors2 = np.array([self._img2_dsc[m[0].trainIdx] for m in self._matches])


        else:
            matches = self.matcher.knnMatch(self._img1_dsc, self._img2_dsc, k=1)
            self._matches = matches
            self._filtered_matches = matches

        self.cache[self._file_index][MATCHES] = self._matches
        self.cache[self._file_index][FILTERED_MATCHES] = self._filtered_matches


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
        self._filtered_matches = filtered
        self.cache[self._file_index][FILTERED_MATCHES] = filtered

    def filter_matches(self, indices, file_index):
        """
        Keeps only matches with given indices.
        :param indices:
        :return:
        """
        matches = []
        for ind in indices:
            matches.append(self._matches[ind])
        self._matches = matches
        self.cache[file_index][MATCHES] = self._matches

    def get_matcher_cache(self):
        self._convert_cache_for_serialization()
        return self.cache

    @staticmethod
    def apply_thresholds(matches, threshold):
        filtered = []
        for i, (m,n) in enumerate(matches):
            if m.distance < threshold * n.distance:
                filtered.append([m])
        return filtered


    def match_between_consecutive_frames(self, prev_frame_index, cur_frame_index, thresh=0.6, debug=False):
        """
        given to frames indices, this function will compute their matching points and cache it.
        :param prev_frame_index:
        :param cur_frame_index:
        :return:
        """

        prev_kps, prev_dsc = self.cache[prev_frame_index][LEFT] # Returns kp, dsc of previous left image
        cur_kps, cur_dsc = self.cache[cur_frame_index][LEFT] # Returns kp, dsc of current left image
        prev_im1 = self.cache[prev_frame_index][FRAMES][0]
        cur_im1 = self.cache[cur_frame_index][FRAMES][0]
        matches = self.matcher.knnMatch(prev_dsc, cur_dsc, k=2)
        # filtered = Matcher.apply_thresholds(matches, thresh)
        self.cache[cur_frame_index][CONSECUTIVE] = filtered
        if debug:
            draw_matches(filtered, prev_im1, cur_im1, prev_kps, cur_kps, num_of_matches=5000, debug=debug, display=VERTICAL_REPRESENTATION)
        return filtered

    def match_between_any_frames(self, reference_frame, other_frame, thresh=0.6, debug=False):
        """
        given to frames indices, this function will compute their matching points and cache it.
        :param thresh: threshold to be applied between frames for KNN
        :param reference_frame:
        :param other_frame:
        :return:
        """

        prev_kps, reference_dsc = self.cache[reference_frame][LEFT] # Returns kp, dsc of previous left image
        cur_kps, other_dsc = self.cache[other_frame][LEFT] # Returns kp, dsc of current left image
        reference_frame_img = self.cache[reference_frame][FRAMES][0]
        other_frame_img = self.cache[other_frame][FRAMES][0]
        matches = self.matcher.knnMatch(reference_dsc, other_dsc, k=2)
        filtered = Matcher.apply_thresholds(matches, thresh)
        if debug:
            draw_matches(filtered, reference_frame_img, other_frame_img, prev_kps, cur_kps, num_of_matches=5000,
                         debug=debug, display=VERTICAL_REPRESENTATION)
        return filtered

    def set_cache(self, matcher_cache):
        self.cache = matcher_cache

    def _convert_cache_for_serialization(self):
        for frame_id in range(2560):
            cur_kps, cur_dsc = self.cache[frame_id][LEFT]
            ser_kps=[]
            for kp in cur_kps:
                ser_kps.append(self.keypoint_to_dict(kp))
            self.cache[frame_id][LEFT] = ser_kps, cur_dsc
        for frame_id in range(2560):
            cur_kps, cur_dsc = self.cache[frame_id][RIGHT]
            ser_kps=[]
            for kp in cur_kps:
                ser_kps.append(self.keypoint_to_dict(kp))
            self.cache[frame_id][RIGHT] = ser_kps, cur_dsc

        for frame_id in range(2560):
            matches = self.cache[frame_id][MATCHES]
            filter_matches = self.cache[frame_id][FILTERED_MATCHES]
            self.cache[frame_id][CONSECUTIVE] = None
            ser_matches=[]
            for match in matches:
                ser_match = self.dmatch_to_dict(match)
                ser_matches.append(ser_match)
            self.cache[frame_id][MATCHES] = ser_matches
            serialized_filter_matches=[]
            for match in filter_matches:
                ser_match = self.dmatch_to_dict(match)
                serialized_filter_matches.append(ser_match)

            self.cache[frame_id][FILTERED_MATCHES] = serialized_filter_matches




    def keypoint_to_dict(self, keypoint):
        return {
            'pt': keypoint.pt,
            'size': keypoint.size,
            'angle': keypoint.angle,
            'response': keypoint.response,
            'octave': keypoint.octave,
            'class_id': keypoint.class_id
        }

    # Convert cv2.DMatch objects to dictionaries
    def dmatch_to_dict(self, dmatch):
        return {'queryIdx': dmatch[0].queryIdx, 'trainIdx': dmatch[0].trainIdx, 'distance': dmatch[0].distance}
