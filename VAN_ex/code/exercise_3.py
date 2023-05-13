import random

import cv2
import numpy as np
from VAN_ex.code.exercise_2 import least_squares
from models.Matcher import Matcher
from utils import utils
from utils.plotters import draw_3d_points, draw_inlier_and_outlier_matches,draw_matches
from utils.utils import rectificatied_stereo_pattern, coords_from_kps, array_to_dict, read_images


cache = {}
matcher = Matcher()
KPS = 0
MATCHES = 1
DSC = 2
HORIZONTAL_REPRESENTATION = 0
VERTICAL_REPRESENTATION = 1

def run_before(lastfunc, *args1, **kwargs1):
    """
    This is a wrapper function which decorates each test, and suppose to serve as a "Garbage Collector", in order to
    clear the current test leftovers, so it won't interfere with the succeeding tests.
    :param lastfunc: usually a testing function
    :param args1:
    :param kwargs1:
    :return:
    """

    def run(func):
        def wrapped_func(*args, **kwargs):
            global result
            try:
                func(*args, **kwargs)
            except AssertionError as e:
                raise e
            finally:
                lastfunc(*args1, **kwargs1)
        return wrapped_func
    return run


def rodriguez_to_mat(rvec, tvec):
  rot, _ = cv2.Rodrigues(rvec)
  return np.hstack((rot, tvec))


def get_3d_points_cloud(matcher, file_index=0, debug=False):
    """
    Computes the 3D points cloud using triangulation from a pair of rectified stereo images.

    Args:
    - matcher: An instance of a Matcher class that has the following methods:
        - detect_and_compute(file_index): Detect and compute the keypoints and descriptors for the input image
        - find_matching_features(with_significance_test=False): Find matching keypoints between two images
    - file_index (int): Index of the input file. Default is 0.
    - debug (bool): If True, the function will display the 3D points cloud. Default is False.

    Returns:
    - inlier_points_in_3d (list): A list of inlier 3D points in the rectified stereo system.
    """
    matcher.detect_and_compute(file_index)
    matcher.find_matching_features(with_significance_test=False)
    matches = matcher.get_matches()
    kp1, kp2 = matcher.get_kp()
    x1, y1, x2, y2, indices_mapping = utils.coords_from_kps(matches, kp1, kp2)
    # Apply rectified stereo pattern on the matches
    img1in, img2in, img1out, img2out = rectificatied_stereo_pattern(y1, y2, indices_mapping, thresh=1)

    x1_in, y1_in = np.array([kp1[idx].pt for idx in img1in]).T
    x2_in, y2_in = np.array([kp2[idx].pt for idx in img2in]).T

    k, m1, m2 = utils.read_cameras()
    inlier_points_in_3d = []
    for idx in range(len(x1_in)):
        our_sol = least_squares((x1_in[idx], y1_in[idx]), (x2_in[idx], y2_in[idx]), k @ m1, k @ m2)
        inlier_points_in_3d.append(our_sol)
    if debug:
        draw_3d_points(inlier_points_in_3d, title=f"3d points from triangulation from image # {file_index}")
    return inlier_points_in_3d



def match_next_pair(cur_file):
    _ = get_3d_points_cloud(matcher, file_index=cur_file, debug=True)
    cur_kps1, cur_kps2 = matcher.get_kp()
    cur_matches = matcher.get_matches()
    return cur_kps1, cur_kps2, cur_matches


def consensus_match(consecutive_matches, prev_indices_mapping, cur_indices_mapping):
    """

    :param consecutive_matches:
    :param prev_indices_mapping:
    :param cur_indices_mapping:
    :return:
    """
    concensus_matces = []
    for idx, (m,n) in enumerate(consecutive_matches):
        prev_left_kp = m.queryIdx
        cur_left_kp = m.trainIdx
        try:
            cur = cur_indices_mapping[cur_left_kp]
            prev = prev_indices_mapping[prev_left_kp]
            concensus_matces.append((prev, cur, idx))
        except KeyError:
            continue
    return concensus_matces

if __name__ == '__main__':
    random.seed(6)

    k, m1, m2 = utils.read_cameras()
    matcher = Matcher(display=HORIZONTAL_REPRESENTATION)

    # Section 3.1
    # Matching features, remove outliers and triangulate
    prev_kps1, prev_kps2, prev_matches = match_next_pair(cur_file=0)
    x1, y1, x2, y2, prev_indices_mapping = coords_from_kps(prev_matches, prev_kps1, prev_kps2)
    img_0_matches = matcher.get_matches()
    prev_indices_mapping = array_to_dict(prev_indices_mapping)

    cur_kps1, cur_kps2, cur_matches = match_next_pair(cur_file=1)
    x1t, y1t, x2t, y2t, cur_indices_mapping = coords_from_kps(cur_matches, cur_kps1, cur_kps2)
    cur_indices_mapping = array_to_dict(cur_indices_mapping)
    img_1_matches = matcher.get_matches()
    # Section 3.2
    # Match features between the two left images (left0 and left1)
    consecutive_matches = matcher.match_between_consecutive_frames(0, 1)

    # Section 3.3
    # Now we that we have:
    #   Matching points in image 0 between left and right
    #   Matching points in image 1 between left and right
    #   Matching points in left frame between image 0 and image 1
    # We can try and match

    # Consensus matches are constructed as follows:
    # each element in the array is a 3-way tuple ->> (img_0_idx, img_1_idx, img0to1_matches_idx)
    consensus_matches = consensus_match(consecutive_matches, prev_indices_mapping, cur_indices_mapping)
    prev, current, consecutive = np.array(consensus_matches)[:,np.arange(3)].T
    xs1_prev, ys1_prev, xs2_prev, ys2_prev, prev_indices_mapping = coords_from_kps(np.array(img_0_matches)[prev], prev_kps1, prev_kps2)

    # define the 3D points in the world coordinate system
    point_3d = []
    # define the corresponding 2D points in the image coordinate system
    xs1, ys1, xs2, ys2, cur_indices_mapping = coords_from_kps(np.array(img_1_matches)[current], cur_kps1, cur_kps2)
    for idx in range(len(xs1)):
        point_3d.append(least_squares((xs1_prev[idx], ys1_prev[idx]), (xs2_prev[idx], ys2_prev[idx]), k @ m1, k @ m2))

    # Choose 4 consensus keypoints and their corresponding 3D points and perform perspective-n-point (PnP)
    # estimation to compute the pose of the camera
    while True:
        idx = np.random.choice(np.arange(4), 4, replace=False)
        success, R_vec, t_vec = cv2.solvePnP(np.array(point_3d,dtype=np.float32)[idx], np.array(cur_indices_mapping,dtype=np.float32).T[idx], k, None, flags=cv2.SOLVEPNP_P3P)
        if success:
            extrinsic_left_camera_matrix_0_to_1 = rodriguez_to_mat(R_vec, t_vec)
            np.vstack(extrinsic_left_camera_matrix_0_to_1, [np.array([0, 0, 0, 1])])
            break

    #TODO: Update the pose of the camera in the world coordinate system
