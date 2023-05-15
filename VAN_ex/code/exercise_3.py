import random

import cv2
import numpy as np
from VAN_ex.code.exercise_2 import least_squares
from models.Matcher import Matcher
from utils import utils
from utils.plotters import draw_3d_points, draw_inlier_and_outlier_matches,draw_matches, plot_four_cameras, draw_supporting_matches, plot_trajectories
from utils.utils import rectificatied_stereo_pattern, coords_from_kps, array_to_dict, read_images
from matplotlib import pyplot as plt
import time



#############################################
################# Constants #################
#############################################
KPS = 0
DSC = 2
HORIZONTAL_REPRESENTATION = 0
VERTICAL_REPRESENTATION = 1
DEFAULT_POV = 0
TOP_POV = 80
SIDE_POV = -120

#############################################

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


def get_3d_points_cloud(inlier_indices_mapping, k, m1, m2, matcher, file_index=0, debug=False):
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


def match_next_pair(cur_file, matcher):
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
    filtered_matches = []
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
            filtered_matches.append(matches_list)
    return concensus_matces, filtered_matches


def solvePnP(kp, corresponding_points, camera_intrinsic, flags=0):
    points_3d = np.array([m[4] for m in corresponding_points])
    points_2d = np.array([kp[m[2]].pt for m in corresponding_points])
    success, R_vec, t_vec = cv2.solvePnP(points_3d, points_2d, camera_intrinsic, None, flags=flags)
    if success:
        camera_extrinsic = rodriguez_to_mat(R_vec, t_vec)
        return camera_extrinsic
    else:
        return None


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
    return are_good_matches, num_good_matches


def ransac_for_pnp(points_to_choose_from, intrinsic_matrix, kp_left, kp_right, right_camera_matrix, thresh=2, debug=False, max_iterations=100):
    num_points_for_model = 4
    best_num_of_supporters = 0
    best_candidate_supporters_boolean_array = []
    epsilon = 0.999
    I = max_iterations
    i = 0
    while i < I:
    #for i in range(max_iterations):
        candidate_4_points = random.sample(points_to_choose_from, k=num_points_for_model)
        candidate_Rt = solvePnP(kp_left, candidate_4_points, intrinsic_matrix, flags=cv2.SOLVEPNP_P3P)
        if candidate_Rt is None:
            i += 1
            continue
        are_supporters_boolean_array, num_good_matches = find_supporters(candidate_Rt, right_camera_matrix, points_to_choose_from, intrinsic_matrix,
                                              kp_left=kp_left, kp_right=kp_right, thresh=thresh, debug=debug)

        if num_good_matches >= best_num_of_supporters:
            best_4_points_candidate = candidate_4_points
            best_candidate_supporters_boolean_array = are_supporters_boolean_array
            best_Rt_candidate = candidate_Rt
            best_num_of_supporters = num_good_matches
            #print(best_num_of_supporters)
        epsilon = min(epsilon, 1 - (num_good_matches / len(are_supporters_boolean_array)))
        I = min(ransac_num_of_iterations(epsilon), max_iterations)
        print(f"at iteration {i} I={I}")
        i += 1
    # We now refine the winner by calculating a transformation for all the supporters/inliers
    supporters = [point_to_choose for ind, point_to_choose in enumerate(points_to_choose_from) if best_candidate_supporters_boolean_array[ind]]
    refined_Rt = solvePnP(kp_left, supporters, intrinsic_matrix, flags=0)
    if refined_Rt is None:
        refined_Rt = best_Rt_candidate
    _, num_good_matches = find_supporters(refined_Rt, right_camera_matrix, points_to_choose_from, intrinsic_matrix,
                                                                     kp_left=kp_left, kp_right=kp_right, thresh=thresh,
                                                                     debug=debug)
    if num_good_matches >= best_num_of_supporters:
        best_Rt_candidate = refined_Rt
        print(f"after refinement: {num_good_matches} supporters")
    return best_Rt_candidate


def apply_Rt_transformation(list_of_3d_points, Rt):
    points_3d_arr = np.array(list_of_3d_points).reshape(3,-1)
    points_3d_arr_hom = np.vstack((points_3d_arr, np.ones(points_3d_arr.shape[1])))
    transformed_points = Rt @ points_3d_arr_hom
    list_of_transformed_points = transformed_points.T.tolist()
    return list_of_transformed_points


def ransac_num_of_iterations(epsilon=0.001, p=0.999, s=4):
    I = np.ceil(np.log(1-p) / np.log(1 - (1 - epsilon) ** s))
    return int(I)

def track_camera_for_many_images(thresh=0.4):
    start_time = time.time()
    k, m1, m2 = utils.read_cameras()
    matcher = Matcher(display=VERTICAL_REPRESENTATION)
    num_of_frames = 2560
    extrinsic_matrices = np.zeros(shape=(num_of_frames, 3, 4))
    camera_positions = np.zeros(shape=(num_of_frames, 3))
    left_camera_extrinsic_mat = m1
    extrinsic_matrices[0] = left_camera_extrinsic_mat
    # initialization
    i = 0
    matcher.read_images(i)
    prev_inlier_indices_mapping = match_next_pair(i, matcher)
    prev_points_cloud, prev_ind_to_3d_point_dict = get_3d_points_cloud(prev_inlier_indices_mapping, k, left_camera_extrinsic_mat, m2, matcher, file_index=i, debug=False)
    prev_indices_mapping = array_to_dict(prev_inlier_indices_mapping)
    # loop over all frames
    for i in range(num_of_frames - 1):
        if i % 10 == 0:
            print(f"----------------- STARTING ITERATION {i} ---------------------")
        cur_inlier_indices_mapping = match_next_pair(i + 1, matcher)
        cur_indices_mapping = array_to_dict(cur_inlier_indices_mapping)
        cur_points_cloud, cur_ind_to_3d_point_dict = get_3d_points_cloud(cur_inlier_indices_mapping, k, left_camera_extrinsic_mat, m2, matcher, file_index=i+1, debug=False)

        consecutive_matches = matcher.match_between_consecutive_frames(i, i + 1, thresh=thresh)
        consensus_matches, filtered_matches = consensus_match(consecutive_matches, prev_indices_mapping, cur_indices_mapping, prev_ind_to_3d_point_dict)

        kp1, kp2 = matcher.get_kp(idx=i + 1)
        Rt = ransac_for_pnp(consensus_matches, k, kp1, kp2, m2, thresh=2,
                            debug=False, max_iterations=500)
        R, t = Rt[:, :-1], Rt[:, -1]
        new_R = R @ extrinsic_matrices[i][:, :-1]
        new_t = R @ extrinsic_matrices[i][:, -1] + t
        new_Rt = np.hstack((new_R, new_t[:, None]))
        extrinsic_matrices[i+1] = new_Rt
        camera_positions[i+1] = -new_R.T @ new_t

        prev_points_cloud, prev_ind_to_3d_point_dict = cur_points_cloud, cur_ind_to_3d_point_dict
        prev_indices_mapping = cur_indices_mapping
    end_time = time.time()
    total_time = end_time - start_time
    print(f"running for {num_of_frames} frames took {total_time} seconds")
    return camera_positions


def get_gt_trajectory():
    gt_extrinsic_matrices = utils.read_gt()
    gt_camera_positions = np.zeros(shape=(gt_extrinsic_matrices.shape[0], 3))
    for i in range(1, gt_camera_positions.shape[0]):
        R_cum, t_cum = gt_extrinsic_matrices[i][:, :-1], gt_extrinsic_matrices[i][:, -1]
        gt_camera_positions[i] = - R_cum.T @ t_cum
    return gt_camera_positions


#------------------------------------------------- EXERCISE QUESTIONS ---------------------------------------

def q1():
    matcher = Matcher(display=VERTICAL_REPRESENTATION)

    prev_inlier_indices_mapping = match_next_pair(0, matcher)
    _, _ = get_3d_points_cloud(prev_inlier_indices_mapping, k, m1, m2, matcher, file_index=0, debug=True)

    cur_inlier_indices_mapping = match_next_pair(1, matcher)
    _, _ = get_3d_points_cloud(cur_inlier_indices_mapping, k, m1, m2, matcher, file_index=1, debug=True)


def q2():

    matcher = Matcher(display=VERTICAL_REPRESENTATION)

    match_next_pair(0, matcher)
    match_next_pair(1, matcher)

    # Matching features between the two consecutive left images (left0 and left1)
    consecutive_matches = matcher.match_between_consecutive_frames(0, 1, thresh=0.4)
    # draw_supporting_matches(1, matcher, consecutive_matches, np.arange(len(consecutive_matches)))


def q3_q4():
    random.seed(6)
    # code from previous questions excluding 3.3
    matcher = Matcher(display=VERTICAL_REPRESENTATION)
    prev_inlier_indices_mapping = match_next_pair(0, matcher)
    prev_points_cloud, prev_ind_to_3d_point_dict = get_3d_points_cloud(prev_inlier_indices_mapping, k, m1, m2, matcher, file_index=0, debug=False)
    prev_indices_mapping = array_to_dict(prev_inlier_indices_mapping)
    cur_inlier_indices_mapping = match_next_pair(1, matcher)
    cur_indices_mapping = array_to_dict(cur_inlier_indices_mapping)
    consecutive_matches = matcher.match_between_consecutive_frames(0, 1, thresh=0.4)

    # from here, new code relevant for 3.3-3.4
    consensus_matches, filtered_matches = consensus_match(consecutive_matches, prev_indices_mapping, cur_indices_mapping, prev_ind_to_3d_point_dict)
    print(f"----------looking for supporters for frame 1----------")
    for i in range(1):
        # Choose 4 consensus keypoints and their corresponding 3D points and perform perspective-n-point (PnP)
        # estimation to compute the pose of the camera
        first_4_consensus_matches = random.sample(consensus_matches, k=4)
        kp1, kp2 = matcher.get_kp(idx=1)
        Rt = solvePnP(kp1, first_4_consensus_matches, k, flags=cv2.SOLVEPNP_P3P)
        if Rt is None:
            continue
        plot_four_cameras(Rt, m2)  # plotting the positions of the camera in the 4 images (Section 3.3)
        are_good_matches, num_good_matches = find_supporters(Rt, m2, consensus_matches, k, kp_left=kp1, kp_right=kp2, thresh=2, debug=True, file_index=1)
        draw_supporting_matches(1, matcher, filtered_matches, are_good_matches)  # (Section 3.4)


def q5():
    random.seed(6)
    # code from previous questions excluding 3.5
    matcher = Matcher(display=VERTICAL_REPRESENTATION)
    prev_inlier_indices_mapping = match_next_pair(0, matcher)
    prev_points_cloud, prev_ind_to_3d_point_dict = get_3d_points_cloud(prev_inlier_indices_mapping, k, m1, m2, matcher, file_index=0, debug=False)
    prev_indices_mapping = array_to_dict(prev_inlier_indices_mapping)
    cur_inlier_indices_mapping = match_next_pair(1, matcher)
    cur_points_cloud, cur_ind_to_3d_point_dict = get_3d_points_cloud(cur_inlier_indices_mapping, k, m1, m2, matcher, file_index=1, debug=False)
    cur_indices_mapping = array_to_dict(cur_inlier_indices_mapping)
    consecutive_matches = matcher.match_between_consecutive_frames(0, 1, thresh=0.4)
    consensus_matches, filtered_matches = consensus_match(consecutive_matches, prev_indices_mapping, cur_indices_mapping, prev_ind_to_3d_point_dict)

    # from here, new code relevant for 3.5

    # # We now use the Ransac frame work to optimize the choice of Rt on given 2 sets of images
    kp1, kp2 = matcher.get_kp(idx=1)
    Rt = ransac_for_pnp(consensus_matches, k, kp1, kp2, m2, thresh=2,
                        debug=False, max_iterations=10)
    # plotting the point clouds for the first two frames
    # transforming the first point cloud according to the Rt we found
    prev_points_cloud = np.array(prev_points_cloud).T
    cur_points_cloud = np.array(cur_points_cloud).T

    prev_points_cloud = prev_points_cloud[:, prev_points_cloud[2] > 0]
    cur_points_cloud = cur_points_cloud[:, cur_points_cloud[2] > 0]

    transformed_point_cloud = apply_Rt_transformation(prev_points_cloud, Rt)
    cur_points_cloud = cur_points_cloud.T.tolist()
    draw_3d_points(transformed_point_cloud,
                   title=f"3d points after transforming from image #0 [Blue] to image #1 [Red]",
                   other_points=cur_points_cloud, pov=DEFAULT_POV)
    draw_3d_points(transformed_point_cloud,
                   title=f"3d points after transforming from image #0 [Blue] to image #1 [Red]",
                   other_points=cur_points_cloud, num_points=np.inf, pov=SIDE_POV)
    draw_3d_points(transformed_point_cloud,
                   title=f"3d points after transforming from image #0 [black] to image #1 [Red]",
                   other_points=cur_points_cloud, num_points=np.inf, pov=TOP_POV)

    are_good_matches, num_good_matches = find_supporters(Rt, m2, consensus_matches, k, kp_left=kp1, kp_right=kp2,
                                                         thresh=2, debug=True, file_index=1)
    draw_supporting_matches(1, matcher, filtered_matches, are_good_matches)


def q6():
    camera_positions = track_camera_for_many_images()
    gt_camera_positions = get_gt_trajectory()
    plot_trajectories(camera_positions, gt_camera_positions)


if __name__ == '__main__':
    random.seed(6)
    k, m1, m2 = utils.read_cameras()
    q1()
    q2()
    q3_q4()
    q5()
    q6()