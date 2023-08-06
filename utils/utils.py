import os
import random

import cv2
import numpy as np
import time
import tqdm
import gtsam
from tqdm import tqdm

from models.Constants import VERTICAL_REPRESENTATION, SEED
from models.TrackDatabase import TrackDatabase
random.seed(SEED)
# from plotters import draw_3d_points, plot_regions_around_matching_pixels

MAC_OS_PATH = "VAN_ex/dataset/sequences/05/"
WINDOWS_OS_PATH = "VAN_ex/dataset/sequences/05\\"
SEP = "\\" if os.name == 'nt' else "/"
DATA_PATH = WINDOWS_OS_PATH if os.name == 'nt' else MAC_OS_PATH

WINDOWS_GT_PATH = "VAN_ex/dataset/poses/05.txt"


def read_images(idx) -> (np.ndarray, np.ndarray):
    """
    Read two images (img1 and img2) given an index and assign them to the Matcher object.
    :param idx: An integer file index for the images to match.
    :return: None
    """
    import cv2
    img_name = '{:06d}.png'.format(idx)
    _img1 = cv2.imread(DATA_PATH + f'image_0{SEP}' + img_name, 0)
    _img2 = cv2.imread(DATA_PATH + f'image_1{SEP}' + img_name, 0)

    return _img1, _img2


def read_gt():
    data = np.loadtxt(WINDOWS_GT_PATH)
    return data.reshape(data.shape[0], 3, 4)

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
    inlier_indices_mapping = indices_mapping[:, inliers_idx]
    img1in, img2in = inlier_indices_mapping[0], inlier_indices_mapping[1]
    img1out, img2out = indices_mapping[0, outliers_idx], indices_mapping[1, outliers_idx]
    return img1in, img2in, img1out, img2out, inlier_indices_mapping


def invert_Rt_transformation(original_Rt):
    R_m = original_Rt[:, :3]
    R = R_m.T
    t_v = original_Rt[:, 3]
    t = - R_m.T @ t_v
    ex_cam_mat_from_cam_to_world = np.hstack((R, t.reshape(3, 1)))
    return ex_cam_mat_from_cam_to_world


def get_gtsam_calib_mat(k, m2):
    """
    Creates a gtsam.Cal3_S2Stereo camera calibration matrix from a given calibration camera matrix.

    Args:
        k (numpy.ndarray): Calibration camera matrix of shape (3, 4).

    Returns:
        gtsam.Cal3_S2Stereo: GTSAM camera calibration matrix.

    """

    # Extract individual calibration parameters from the calibration camera matrix
    fx = k[0, 0]   # Focal length in x-axis
    fy = k[1, 1]   # Focal length in y-axis
    skew = k[0, 1]  # Skew
    cx = k[0, 2]   # Principal point x-coordinate
    cy = k[1, 2]   # Principal point y-coordinate

    # Get baseline from some data source (e.g., Data.KITTI.get_M2())
    baseline = m2[0, 3]

    # Create a GTSAM Cal3_S2Stereo camera calibration matrix
    gtsam_calib_mat = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)

    return gtsam_calib_mat


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

import cv2
import numpy as np

def select_good_matches(keypoints1, keypoints2, matches, reprojection_threshold=3.0, inlier_ratio_threshold=0.1):
    # Convert keypoints to numpy arrays
    keypoints1_np = np.array([kp.pt for kp in keypoints1], dtype=np.float32)
    keypoints2_np = np.array([kp.pt for kp in keypoints2], dtype=np.float32)

    # Convert matches to arrays of point indices
    src_pts = np.array([keypoints1_np[m[0].queryIdx] for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([keypoints2_np[m[0].trainIdx] for m in matches], dtype=np.float32).reshape(-1, 1, 2)

    # Estimate fundamental matrix using RANSAC
    _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, reprojection_threshold)
    k, m1, m2 = read_cameras()
    disparities = np.abs(src_pts[:, 0, 0] - dst_pts[:, 0, 0])

    disparity_thresh = np.median(disparities)

    # Compute reprojection error for inlier matches
    reprojection_errors = []
    inlier_matches = []
    for i, m in enumerate(matches):
        if mask[i] == 1:
            src_pt = src_pts[i, 0]
            dst_pt = dst_pts[i, 0]
            if abs(src_pt[1]-dst_pt[1]) > 1:
                print(src_pt[1]-dst_pt[1])
                continue
            # F = cv2.findFundamentalMat(np.array([src_pt]), np.array([dst_pt]), cv2.FM_8POINT)[0]
            pts_3d = cv2.triangulatePoints(k @ m1, k @ m2, dst_pt,src_pt)
            pts_3d /= pts_3d[3]
            reprojection_error = np.linalg.norm(pts_3d[:2] - dst_pt)
            reprojection_errors.append(reprojection_error)
            inlier_matches.append(m)
    reprojection_error_threshold = np.median(reprojection_errors)
    # Compute inlier ratio

    # Select good matches based on reprojection error and inlier ratio thresholds
    good_matches = []
    for error, match, disparity in zip(reprojection_errors, inlier_matches, disparities):
        if error <= reprojection_error_threshold and disparity < disparity_thresh:
            good_matches.append(match)

    return good_matches


def find_all_minima(points_array: list):
    """
    Given an array of integers, this function will find all the minima points indices
    @:param: points_array: a list containing the number of 1D points
    :return: The index of all minima points in the given input
    """
    minimas = np.array(points_array)
    # Finds all local minima
    return np.where((minimas[1:-1] < minimas[0:-2]) * (
            minimas[1:-1] < minimas[2:]))[0]


def track_camera_for_many_images(thresh=1):
    from models.Matcher import Matcher

    k, m1, m2 = read_cameras()
    matcher = Matcher(display=VERTICAL_REPRESENTATION)
    num_of_frames = 2560
    extrinsic_matrices = np.zeros(shape=(num_of_frames, 3, 4))
    camera_positions = np.zeros(shape=(num_of_frames, 3))
    left_camera_extrinsic_mat = m1
    extrinsic_matrices[0] = left_camera_extrinsic_mat

    # Initialize track database
    track_db = TrackDatabase()

    # Initialization
    frameId = 0
    matcher.read_images(frameId)
    prev_inlier_indices_mapping = match_next_pair(frameId, matcher)
    prev_points_cloud, prev_ind_to_3d_point_dict = get_3d_points_cloud(prev_inlier_indices_mapping, k,
                                                                       left_camera_extrinsic_mat, m2, matcher,
                                                                       file_index=frameId, debug=False)
    prev_indices_mapping = array_to_dict(prev_inlier_indices_mapping)

    # Loop over all frames
    for frameId in tqdm(range(num_of_frames-1)):
        cur_inlier_indices_mapping = match_next_pair(frameId + 1, matcher)
        cur_indices_mapping = array_to_dict(cur_inlier_indices_mapping)
        cur_points_cloud, cur_ind_to_3d_point_dict = get_3d_points_cloud(cur_inlier_indices_mapping, k,
                                                                         left_camera_extrinsic_mat, m2, matcher,
                                                                         file_index=frameId + 1, debug=False)

        consecutive_matches = matcher.match_between_consecutive_frames(frameId, frameId + 1, thresh=thresh)

        # UPDATED: Note that the consensus_matche function is modified to work with the tracking DB mechanism (!)
        matches_that_appear_in_all_4_images, filtered_matches = get_matches_that_appear_in_all_4_images(consecutive_matches, prev_indices_mapping,
                                                                                      cur_indices_mapping, prev_ind_to_3d_point_dict)

        kp1, kp2 = matcher.get_kp(idx=frameId + 1)
        Rt, inliers_ratio, are_supporters_boolean_array = ransac_for_pnp(matches_that_appear_in_all_4_images, k, kp1, kp2, m2, thresh=2,
                            debug=False, max_iterations=1000, return_supporters=True)
        assert are_supporters_boolean_array is not None, f"RANSAC Couldn't find a transformation from frame {frameId} to frame {frameId + 1}..."
        Rt_inliers = [point_to_choose for ind, point_to_choose in enumerate(matches_that_appear_in_all_4_images) if
                  are_supporters_boolean_array[ind]]
        num_inliers = np.sum(are_supporters_boolean_array)
        track_db.update_tracks_from_frame(matcher, frameId, Rt_inliers)
        track_db.add_inliers_ratio(frameId, inliers_ratio)
        track_db.add_num_matches(frameId, num_matches=num_inliers)
        R, t = Rt[:, :-1], Rt[:, -1]
        new_R = R @ extrinsic_matrices[frameId][:, :-1]
        new_t = R @ extrinsic_matrices[frameId][:, -1] + t
        new_Rt = np.hstack((new_R, new_t[:, None]))
        extrinsic_matrices[frameId + 1] = new_Rt
        camera_positions[frameId + 1] = -new_R.T @ new_t
        prev_points_cloud, prev_ind_to_3d_point_dict = cur_points_cloud, cur_ind_to_3d_point_dict
        prev_indices_mapping = cur_indices_mapping
        track_db.set_ex_camera_positions(camera_positions)
        track_db.set_ex_matrices(extrinsic_matrices)

    return camera_positions, track_db


def match_next_pair(cur_file, matcher):
    inlier_indices_mapping = find_stereo_matches(matcher, cur_file)
    return inlier_indices_mapping


def find_stereo_matches(matcher, file_index):
    matcher.detect_and_compute(file_index)
    matcher.find_matching_features(with_significance_test=False)
    matches = matcher.get_matches(file_index)
    filtered_matches = matcher.get_filtered_matches(file_index)

    kp1, kp2 = matcher.get_kp(file_index)

    x1, y1, x2, y2, indices_mapping = coords_from_kps(filtered_matches, kp1, kp2)
    # Apply rectified stereo pattern on the matches
    img1in, img2in, img1out, img2out, inlier_indices_mapping = rectificatied_stereo_pattern(y1, y2, indices_mapping,
                                                                                            thresh=2)
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
        if our_sol[-1] <= 0 or x1 < x2:
            continue

        if abs(y1-y2)>2:
            print(f"y1-y2: {y1-y2} for y1: {y1}, y2:{y2}")
        inlier_points_in_3d.append(our_sol)
        ind_to_3d_point_dict[ind_1] = our_sol
    if debug:
        draw_3d_points(inlier_points_in_3d, title=f"3d points from triangulation from image # {file_index}")
    return inlier_points_in_3d, ind_to_3d_point_dict


def get_matches_that_appear_in_all_4_images(consecutive_matches, prev_indices_mapping, cur_indices_mapping, ind_to_3d_point_prev):
    """
    * Given prev_left_ind, prev_right_ind, cur_left_ind, cur_right_ind
    ** Consider the first iteration, i.e., the consensus match between the 0th and the 1st frames:
    **** FrameId is the 0 index (TODO: We should think about which of the two frames indices is more meaningful for this purpose)
    **** Every consensus match is assigned as a new track with length of 2.
    **** We start by appending the prev_left_ind, cur_left_ind to each track
    **** each track is composed by 3 elements: (kp_prev, feature_location_prev, frameId)
    **** In order to manage the bookkeeping of the tracks, we use a dictionary that is constructed as follows:
    **** A double dictionary to store last left_kps indices (Keys) and a matching trackId (Values)
    *********** [See _last_insertions in TrackDatabase]

    **** Each matching between the pair of consecutive frames is keeping, where each kp used as a key, and the trackId
    **** is kept as its value, where the primary key is the frameId.
    **** Another thing we need to consider is that each time we insert new keypoint (calling add_track in other words)
    **** we update the _last_insertions for the next iteration by assigning the current left kp (cur_left_kp).
    **** on the i'th frameId, we update the _last_insertions[(i+1)%2], where _last_insertions is a dictionary of 2
    **** dictionaries we constantly update.

    ** 2+ iterations
    **** Then, from this stage, we shall expect existing tracks in our db.
    **** So, for any feature (referred as prev_left_ind) which is already belong to an existing track, we should be
    **** able to find it in the _last_insertions[frameId][prev_left_ind], since on last iteration we inserted the
    **** corresponding keypoint - cur_left_kp to the _last_insertions[(i+1) % 2] dictionary.

    :param matcher1:
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
        if prev_left_kp in prev_indices_mapping and cur_left_kp in cur_indices_mapping and prev_left_kp in ind_to_3d_point_prev:
            prev_left_ind = prev_left_kp
            assert prev_left_ind in ind_to_3d_point_prev, f"prev_left_ind={prev_left_ind} not in ind_to_3d_point_prev for some reason..."
            prev_left_ind = prev_left_kp
            prev_right_ind = prev_indices_mapping[prev_left_kp]
            cur_left_ind = cur_left_kp
            cur_right_ind = cur_indices_mapping[cur_left_kp]
            point_3d_prev = ind_to_3d_point_prev[prev_left_ind]
            concensus_matces.append((prev_left_ind, prev_right_ind, cur_left_ind, cur_right_ind, point_3d_prev))
            filtered_matches.append(matches_list)
    return concensus_matces, filtered_matches


def find_supporters(Rt, m2, consensus_matches, k, kp_left, kp_right, thresh=2, debug=True, file_index=0):
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


def ransac_for_pnp(points_to_choose_from, intrinsic_matrix, kp_left, kp_right, right_camera_matrix, thresh=2,
                   debug=False, max_iterations=100, return_supporters=False):

    num_points_for_model = 4
    if len(points_to_choose_from) < num_points_for_model:
        print("too small population")
        return (None, None, None) if return_supporters else (None, None)
    best_num_of_supporters = 0
    best_candidate_supporters_boolean_array = []
    epsilon = 0.99
    I = max_iterations
    i = 0
    while i < I and epsilon != 0:
        # for i in range(max_iterations):
        candidate_4_points = random.sample(points_to_choose_from, k=num_points_for_model)
        candidate_Rt = solvePnP(kp_left, candidate_4_points, intrinsic_matrix, flags=cv2.SOLVEPNP_AP3P)
        if candidate_Rt is None:
            #print(i)
            i += 1
            continue
        are_supporters_boolean_array, num_good_matches = find_supporters(candidate_Rt, right_camera_matrix,
                                                                         points_to_choose_from, intrinsic_matrix,
                                                                         kp_left=kp_left, kp_right=kp_right,
                                                                         thresh=thresh, debug=debug)

        if num_good_matches > best_num_of_supporters:
            best_4_points_candidate = candidate_4_points
            best_candidate_supporters_boolean_array = are_supporters_boolean_array
            best_Rt_candidate = candidate_Rt
            best_num_of_supporters = num_good_matches
        epsilon = min(epsilon, 1 - (num_good_matches / len(are_supporters_boolean_array)))
        I = min(ransac_num_of_iterations(epsilon, 0.999), max_iterations)
        #print(f"at iteration {i} I={I}")
        i += 1
    # We now refine the winner by calculating a transformation for all the supporters/inliers
    if len(best_candidate_supporters_boolean_array) == 0:
        return (None, None, None) if return_supporters else (None, None)
    supporters = [point_to_choose for ind, point_to_choose in enumerate(points_to_choose_from) if
                  best_candidate_supporters_boolean_array[ind]]
    refined_Rt = solvePnP(kp_left, supporters, intrinsic_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
    if refined_Rt is None:
        refined_Rt = best_Rt_candidate
    are_supporters_boolean_array, num_good_matches = find_supporters(refined_Rt, right_camera_matrix, points_to_choose_from, intrinsic_matrix,
                                          kp_left=kp_left, kp_right=kp_right, thresh=thresh,
                                          debug=debug)
    if num_good_matches >= best_num_of_supporters:
        best_Rt_candidate = refined_Rt
        print(f"after refinement: {num_good_matches} supporters out of {len(points_to_choose_from)} matches")
    inliers_ratio = num_good_matches / len(points_to_choose_from)
    if return_supporters:
        return best_Rt_candidate, inliers_ratio, are_supporters_boolean_array
    return best_Rt_candidate, inliers_ratio


def apply_Rt_transformation(list_of_3d_points, Rt):
    points_3d_arr = np.array(list_of_3d_points).reshape(3, -1)
    points_3d_arr_hom = np.vstack((points_3d_arr, np.ones(points_3d_arr.shape[1])))
    transformed_points = Rt @ points_3d_arr_hom
    list_of_transformed_points = transformed_points.T.tolist()
    return list_of_transformed_points


def ransac_num_of_iterations(epsilon=0.99, p=0.99, s=4):
    if epsilon == 0:
        return 0
    return int(np.ceil(np.log(1 - p) / np.log(1 - (1 - epsilon) ** s)))


def solvePnP(kp, corresponding_points, camera_intrinsic, flags=0):
    points_3d = np.array([m[4] for m in corresponding_points])
    points_2d = np.array([kp[m[2]].pt for m in corresponding_points])
    try:
        success, R_vec, t_vec = cv2.solvePnP(points_3d, points_2d, camera_intrinsic, None, flags=flags)
        if success:
            camera_extrinsic = rodriguez_to_mat(R_vec, t_vec)
            return camera_extrinsic
        else:
            return None
    except:
        return None


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def project_point_on_image(point_3d, extrinsic_matrix, intrinsic_matrix):
    point_3d_hom = np.r_[point_3d, 1]
    projected_point_hom = intrinsic_matrix @ extrinsic_matrix @ point_3d_hom
    projected_point = projected_point_hom[:2] / projected_point_hom[2]
    return projected_point


def visualize_track(track, num_to_show=10):
    from utils.plotters import plot_regions_around_matching_pixels
    first_frames_of_track = track[:num_to_show]
    for i, track_point in enumerate(first_frames_of_track):

        _, feature_location, frameId = track_point
        x_l, x_r, y = feature_location
        left_image, right_image = read_images(frameId)
        plot_regions_around_matching_pixels(left_image, right_image, x_l, y, x_r, y, frame_index=frameId, path=f"plots/debug_track{frameId}")


def get_gt_trajectory():
    gt_extrinsic_matrices = read_gt()
    gt_camera_positions = np.zeros(shape=(gt_extrinsic_matrices.shape[0], 3))
    for i in range(1, gt_camera_positions.shape[0]):
        R_cum, t_cum = gt_extrinsic_matrices[i][:, :-1], gt_extrinsic_matrices[i][:, -1]
        gt_camera_positions[i] = - R_cum.T @ t_cum
    return gt_camera_positions


def least_squares(p1, p2, Pmat, Qmat):
    """
    Least Squares algorithm as learned in class
    :param p1: (x,y) Point in the left image
    :param p2: (x,y) Point in the right image
    :param Pmat: Left image matrix
    :param Qmat: Right image matrix
    :return:
    """
    x1, y1 = p1
    x2, y2 = p2

    P1 = Pmat[0, :]
    P2 = Pmat[1, :]
    P3 = Pmat[2, :]
    Q1 = Qmat[0, :]
    Q2 = Qmat[1, :]
    Q3 = Qmat[2, :]

    # Computing the 'A' matrix from the solution of Ax=0
    A = np.array([x1 * P3 - P1,
         y1 * P3 - P2,
         x2 * Q3 - Q1,
         y2 * Q3 - Q2])

    # Compute the SVD solution for A matrix
    u, s, vh = np.linalg.svd(A)

    # We saw in class that the solution is located in the last column of vh:
    solution_4d = vh[-1]

    # Transform from 4d to 3d:
    solution_3d = solution_4d[:3] / solution_4d[-1]
    return solution_3d
