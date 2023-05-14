import random

import cv2
import numpy as np
from VAN_ex.code.exercise_2 import least_squares
from models.Matcher import Matcher
from utils import utils
from utils.plotters import draw_3d_points, draw_inlier_and_outlier_matches,draw_matches, plot_four_cameras, draw_supporting_matches
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


def find_stereo_matches(matcher, file_index, debug=False):
    matcher.detect_and_compute(file_index)
    matcher.find_matching_features(with_significance_test=False)
    matches = matcher.get_matches(file_index)
    kp1, kp2 = matcher.get_kp(file_index)
    x1, y1, x2, y2, indices_mapping = utils.coords_from_kps(matches, kp1, kp2)
    # Apply rectified stereo pattern on the matches
    img1in, img2in, img1out, img2out, inlier_indices_mapping = rectificatied_stereo_pattern(y1, y2, indices_mapping, thresh=1)
    matcher.filter_matches(img1in, file_index)
    return inlier_indices_mapping


def get_3d_points_cloud(inlier_indices_mapping, k, m1, m2, file_index=0, debug=False):
    kp1, kp2 = matcher.get_kp(file_index)
    inlier_points_in_3d = []
    ind_to_3d_point_dict = dict()
    for ind_1, ind_2 in inlier_indices_mapping.T:
        x1, y1 = kp1[ind_1].pt
        x2, y2 = kp2[ind_2].pt
        our_sol = least_squares((x1, y1), (x2, y2), k @ m1, k @ m2)
        inlier_points_in_3d.append(our_sol)
        ind_to_3d_point_dict[ind_1] = our_sol
    if debug:
        draw_3d_points(inlier_points_in_3d, title=f"3d points from triangulation from image # {file_index}")
    return inlier_points_in_3d, ind_to_3d_point_dict


def get_3d_points_cloud_orig(matcher, file_index=0, debug=False):
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
    inlier_indices_mapping = find_stereo_matches(matcher, cur_file, debug=False)
    return inlier_indices_mapping


def consensus_match(consecutive_matches, prev_indices_mapping, cur_indices_mapping, ind_to_3d_point_prev):
    """

    :param consecutive_matches:
    :param prev_indices_mapping:
    :param cur_indices_mapping:
    :return:
    """
    concensus_matces = []
    for idx, matches_list in enumerate(consecutive_matches):
        m = matches_list[0]
        prev_left_kp = m.queryIdx
        cur_left_kp = m.trainIdx
        if prev_left_kp in prev_indices_mapping and cur_left_kp in cur_indices_mapping:
            prev_left_ind = prev_left_kp
            prev_right_ind = prev_indices_mapping[prev_left_kp]
            cur_left_ind = cur_left_kp
            cur_right_ind = cur_indices_mapping[cur_left_kp]
            point_3d_prev = ind_to_3d_point_prev[prev_left_ind]
            concensus_matces.append((prev_left_ind, prev_right_ind, cur_left_ind, cur_right_ind, point_3d_prev))
    return concensus_matces


def solvePnP(kp, corresponding_points, camera_intrinsic):
    points_3d = np.array([m[4] for m in corresponding_points])
    points_2d = np.array([kp[m[2]].pt for m in corresponding_points])
    success, R_vec, t_vec = cv2.solvePnP(points_3d, points_2d, camera_intrinsic, None, flags=cv2.SOLVEPNP_P3P)
    camera_extrinsic = rodriguez_to_mat(R_vec, t_vec)
    return camera_extrinsic


def find_supporters(Rt, m2, consensus_matches, k, kp_left, kp_right, thresh=2, debug=True, file_index = 0):
    are_good_matches = np.zeros(len(consensus_matches))
    num_good_matches = 0
    for i, m in enumerate(consensus_matches):
        prev_left_ind, prev_right_ind, cur_left_ind, cur_right_ind, point_3d_prev = m
        point_in_prev_coordinates_hom = np.r_[point_3d_prev, 1]
        left_1_3d_location = Rt @ point_in_prev_coordinates_hom
        right_1_3d_location = left_1_3d_location + m2[:, -1]
        left_1_2d_location_projected_hom = k @ left_1_3d_location
        left_1_2d_location_projected = left_1_2d_location_projected_hom[:2] / left_1_2d_location_projected_hom[2]
        right_1_2d_location_projected_hom = k @ right_1_3d_location
        right_1_2d_location_projected = right_1_2d_location_projected_hom[:2] / right_1_2d_location_projected_hom[2]
        left_1_2d_location_real = np.array(kp_left[cur_left_ind].pt)
        right_1_2d_location_real = np.array(kp_right[cur_right_ind].pt)
        left_dist = np.linalg.norm(left_1_2d_location_real - left_1_2d_location_projected)
        right_dist = np.linalg.norm(right_1_2d_location_real - right_1_2d_location_projected)
        is_supporter = (left_dist < thresh) and (right_dist < thresh)
        are_good_matches[i] = is_supporter
        num_good_matches += is_supporter
    if debug:
        print(f"out of {len(consensus_matches)} matches, {num_good_matches} are supporters for this Rt choice")
    return are_good_matches


def ransac_for_pnp():
    ...


if __name__ == '__main__':
    random.seed(6)

    k, m1, m2 = utils.read_cameras()
    matcher = Matcher(display=VERTICAL_REPRESENTATION)

    # ------------------------------------------------Section 3.1-----------------------------------------------

    # Matching features in stereo images and creating points clouds, both in frame 0 and in frame 1
    prev_inlier_indices_mapping = match_next_pair(cur_file=0)
    prev_points_cloud, prev_ind_to_3d_point_dict = get_3d_points_cloud(prev_inlier_indices_mapping, k, m1, m2, file_index=0, debug=False)
    prev_indices_mapping = array_to_dict(prev_inlier_indices_mapping)
    img_0_matches = matcher.get_matches(idx=0)

    cur_inlier_indices_mapping = match_next_pair(cur_file=1)
    cur_points_cloud, cur_ind_to_3d_point_dict = get_3d_points_cloud(cur_inlier_indices_mapping, k, m1, m2, file_index=1, debug=False)
    cur_indices_mapping = array_to_dict(cur_inlier_indices_mapping)
    img_1_matches = matcher.get_matches(idx=1)

    # ------------------------------------------------Section 3.2-----------------------------------------------

    # Matching features between the two consecutive left images (left0 and left1)
    consecutive_matches = matcher.match_between_consecutive_frames(0, 1, thresh=0.5)

    # -----------------------------------------Section 3.3 + Section 3.4-----------------------------------------

    # Now that we have:
    #   Matching points in image 0 between left and right
    #   Matching points in image 1 between left and right
    #   Matching points in left frame between image 0 and image 1
    # We find the transformation between frame 0 and frame 1 based on that

    # Consensus matches are constructed as follows: each element in the array is a 5-way tuple ->>
    # (left_0_idx, right_0_idx, left_1_idx, right_1_idx, corresponding 3d point based on frame 0)
    consensus_matches = consensus_match(consecutive_matches, prev_indices_mapping, cur_indices_mapping, prev_ind_to_3d_point_dict)
    print(f"----------looking for supporters for frame 1----------")
    for i in range(1):
        # Choose 4 consensus keypoints and their corresponding 3D points and perform perspective-n-point (PnP)
        # estimation to compute the pose of the camera
        first_4_consensus_matches = random.sample(consensus_matches, k=4)
        kp1, kp2 = matcher.get_kp(idx=1)
        Rt = solvePnP(kp1, first_4_consensus_matches, k)
        plot_four_cameras(Rt, m2)  # plotting the positions of the camera in the 4 images (Section 3.3)
        are_good_matches = find_supporters(Rt, m2, consensus_matches, k, kp_left=kp1, kp_right=kp2, thresh=2, debug=True, file_index=1)
        draw_supporting_matches(0, matcher, consensus_matches, are_good_matches)  # (Section 3.4)

    # ------------------------------------------------Section 3.5-----------------------------------------------
    # We now use the Ransac frame work to optimize the choice of Rt on given 2 sets of images
    exit()



