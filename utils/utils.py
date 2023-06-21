import os
import cv2
import numpy as np
import time
import tqdm
import gtsam
MAC_OS_PATH = "../dataset/sequences/05/"
WINDOWS_OS_PATH = "../dataset/sequences/05\\"
SEP = "\\" if os.name == 'nt' else "/"
DATA_PATH = WINDOWS_OS_PATH if os.name == 'nt' else MAC_OS_PATH

WINDOWS_GT_PATH = "../dataset/poses/05.txt"


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


def track_camera_for_many_images(thresh=0.4):
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
    for i in tqdm.tqdm.range(num_of_frames - 1):
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

    return camera_positions

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