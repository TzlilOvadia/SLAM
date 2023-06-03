import random
import cv2
import numpy as np
from tqdm import tqdm

from VAN_ex.code.exercise_2 import least_squares
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from utils import utils
from utils.plotters import draw_3d_points, draw_inlier_and_outlier_matches, draw_matches, plot_four_cameras, \
    draw_supporting_matches, plot_trajectories
from utils.utils import rectificatied_stereo_pattern, coords_from_kps, array_to_dict, read_images
from matplotlib import pyplot as plt
from models.Constants import *


def track_camera_for_many_images(thresh=0.4):
    k, m1, m2 = utils.read_cameras()
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
    for frameId in tqdm(range(num_of_frames - 1)):
        cur_inlier_indices_mapping = match_next_pair(frameId + 1, matcher)
        cur_indices_mapping = array_to_dict(cur_inlier_indices_mapping)
        cur_points_cloud, cur_ind_to_3d_point_dict = get_3d_points_cloud(cur_inlier_indices_mapping, k,
                                                                         left_camera_extrinsic_mat, m2, matcher,
                                                                         file_index=frameId + 1, debug=False)
        consecutive_matches = matcher.match_between_consecutive_frames(frameId, frameId + 1, thresh=thresh)

        # UPDATED: Note that the consensus_matche function is modified to work with the tracking DB mechanism (!)
        consensus_matches, filtered_matches = consensus_match(consecutive_matches, prev_indices_mapping,
                                                              cur_indices_mapping, prev_ind_to_3d_point_dict, track_db,
                                                              frameId, matcher)

        kp1, kp2 = matcher.get_kp(idx=frameId + 1)
        Rt = ransac_for_pnp(consensus_matches, k, kp1, kp2, m2, thresh=2,
                            debug=False, max_iterations=500)
        R, t = Rt[:, :-1], Rt[:, -1]
        new_R = R @ extrinsic_matrices[frameId][:, :-1]
        new_t = R @ extrinsic_matrices[frameId][:, -1] + t
        new_Rt = np.hstack((new_R, new_t[:, None]))
        extrinsic_matrices[frameId + 1] = new_Rt
        camera_positions[frameId + 1] = -new_R.T @ new_t
        prev_points_cloud, prev_ind_to_3d_point_dict = cur_points_cloud, cur_ind_to_3d_point_dict
        prev_indices_mapping = cur_indices_mapping

    return camera_positions, track_db


def match_next_pair(cur_file, matcher):
    inlier_indices_mapping = find_stereo_matches(matcher, cur_file)
    return inlier_indices_mapping


def find_stereo_matches(matcher, file_index):
    matcher.detect_and_compute(file_index)
    matcher.find_matching_features(with_significance_test=False)
    matches = matcher.get_matches(file_index)
    kp1, kp2 = matcher.get_kp(file_index)
    x1, y1, x2, y2, indices_mapping = utils.coords_from_kps(matches, kp1, kp2)
    # Apply rectified stereo pattern on the matches
    img1in, img2in, img1out, img2out, inlier_indices_mapping = rectificatied_stereo_pattern(y1, y2, indices_mapping,
                                                                                            thresh=1)
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


def consensus_match(consecutive_matches, prev_indices_mapping, cur_indices_mapping, ind_to_3d_point_prev,
                    track_db: TrackDatabase, frameId: int, matcher: Matcher):
    """

    :param matcher1:
    :param consecutive_matches:
    :param prev_indices_mapping:
    :param cur_indices_mapping:
    :return:
    """

    # * Given prev_left_ind, prev_right_ind, cur_left_ind, cur_right_ind
    # ** Consider the first iteration, i.e., the consensus match between the 0th and the 1st frames:
    # **** FrameId is the 0 index (TODO: We should think about which of the two frames indices is more meaningful for this purpose)
    # **** Every consensus match is assigned as a new track with length of 2.
    # **** We start by appending the prev_left_ind, cur_left_ind to each track
    # **** each track is composed by 3 elements: (kp_prev, feature_location_prev, frameId)
    # **** In order to manage the bookkeeping of the tracks, we use a dictionary that is constructed as follows:
    # **** A double dictionary to store last left_kps indices (Keys) and a matching trackId (Values)
    # *********** [See _last_insertions in TrackDatabase]

    # **** Each matching between the pair of consecutive frames is keeping, where each kp used as a key, and the trackId
    # **** is kept as its value, where the primary key is the frameId.
    # **** Another thing we need to consider is that each time we insert new keypoint (calling add_track in other words)
    # **** we update the _last_insertions for the next iteration by assigning the current left kp (cur_left_kp).
    # **** on the i'th frameId, we update the _last_insertions[(i+1)%2], where _last_insertions is a dictionary of 2
    # **** dictionaries we constantly update.

    # ** 2+ iterations
    # **** Then, from this stage, we shall expect existing tracks in our db.
    # **** So, for any feature (referred as prev_left_ind) which is already belong to an existing track, we should be
    # **** able to find it in the _last_insertions[frameId][prev_left_ind], since on last iteration we inserted the
    # **** corresponding keypoint - cur_left_kp to the _last_insertions[(i+1) % 2] dictionary.

    concensus_matces = []
    filtered_matches = []
    track_db.prepare_to_next_pair(frameId)

    for idx, matches_list in enumerate(consecutive_matches):
        m = matches_list[0]
        prev_left_kp = m.queryIdx
        cur_left_kp = m.trainIdx

        if prev_left_kp in prev_indices_mapping and cur_left_kp in cur_indices_mapping:
            trackId = track_db.get_kp_trackId(prev_left_kp, frameId)

            #  Every consnsused match is assigned as a new track with length of 2.
            #  We start by appending the prev_left_ind, cur_left_ind to each track
            #  -->> [] is updated to [prev_left_ind, cur_left_ind]

            if trackId is None:
                trackId = track_db.generate_new_track_id()

            xl, yl = matcher.get_feature_location_frame(frameId, kp=prev_left_kp, loc=LEFT)
            xr, yr = matcher.get_feature_location_frame(frameId, kp=prev_indices_mapping[prev_left_kp], loc=RIGHT)

            # xl_n,yl_n = matcher.get_feature_location_frame(frameId+1, kp=prev_left_kp, loc=LEFT)
            # xr_n,yr_n = matcher.get_feature_location_frame(frameId+1, kp=prev_indices_mapping[prev_left_kp], loc=RIGHT)
            feature_location_prev = (xl, xr, yl)
            # feature_location_cur = (xl_n, xr_n, yl_n)

            track_db.add_track(trackId, frameId, feature_location_prev, prev_left_kp, cur_left_kp)

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
                   debug=False, max_iterations=100):
    num_points_for_model = 4
    best_num_of_supporters = 0
    best_candidate_supporters_boolean_array = []
    epsilon = 0.999
    I = max_iterations
    i = 0
    while i < I:
        # for i in range(max_iterations):
        candidate_4_points = random.sample(points_to_choose_from, k=num_points_for_model)
        candidate_Rt = solvePnP(kp_left, candidate_4_points, intrinsic_matrix, flags=cv2.SOLVEPNP_P3P)
        if candidate_Rt is None:
            i += 1
            continue
        are_supporters_boolean_array, num_good_matches = find_supporters(candidate_Rt, right_camera_matrix,
                                                                         points_to_choose_from, intrinsic_matrix,
                                                                         kp_left=kp_left, kp_right=kp_right,
                                                                         thresh=thresh, debug=debug)

        if num_good_matches >= best_num_of_supporters:
            best_4_points_candidate = candidate_4_points
            best_candidate_supporters_boolean_array = are_supporters_boolean_array
            best_Rt_candidate = candidate_Rt
            best_num_of_supporters = num_good_matches
        epsilon = min(epsilon, 1 - (num_good_matches / len(are_supporters_boolean_array)))
        I = min(ransac_num_of_iterations(epsilon), max_iterations)
        print(f"at iteration {i} I={I}")
        i += 1
    # We now refine the winner by calculating a transformation for all the supporters/inliers
    supporters = [point_to_choose for ind, point_to_choose in enumerate(points_to_choose_from) if
                  best_candidate_supporters_boolean_array[ind]]
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
    points_3d_arr = np.array(list_of_3d_points).reshape(3, -1)
    points_3d_arr_hom = np.vstack((points_3d_arr, np.ones(points_3d_arr.shape[1])))
    transformed_points = Rt @ points_3d_arr_hom
    list_of_transformed_points = transformed_points.T.tolist()
    return list_of_transformed_points


def ransac_num_of_iterations(epsilon=0.001, p=0.999, s=4):
    if epsilon == 0:
        return 0
    return int(np.ceil(np.log(1 - p) / np.log(1 - (1 - epsilon) ** s)))


def solvePnP(kp, corresponding_points, camera_intrinsic, flags=0):
    points_3d = np.array([m[4] for m in corresponding_points])
    points_2d = np.array([kp[m[2]].pt for m in corresponding_points])
    success, R_vec, t_vec = cv2.solvePnP(points_3d, points_2d, camera_intrinsic, None, flags=flags)
    if success:
        camera_extrinsic = rodriguez_to_mat(R_vec, t_vec)
        return camera_extrinsic
    else:
        return None


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


if __name__ == "__main__":
    # Create an instance of TrackDatabase
    camera_positions, track_db = track_camera_for_many_images()