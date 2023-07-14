import random
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from scipy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from VAN_ex.code.exercise_2 import least_squares
from VAN_ex.code.exercise_3 import get_gt_trajectory
from VAN_ex.code.exercise_5 import solve_one_bundle, get_bundle_windows, criteria
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from utils import utils
from utils.plotters import draw_3d_points, draw_inlier_and_outlier_matches, draw_matches, plot_four_cameras, \
    draw_supporting_matches, plot_trajectories, plot_regions_around_matching_pixels, plot_dict, plot_connectivity_graph, \
    gen_hist, plot_reprojection_errors, plot_localization_error_over_time, plot_projections_on_images, plot_trajectory_and_points, plot_2d_cameras_and_points
from utils.utils import *
from matplotlib import pyplot as plt
from models.Constants import *
import gtsam
from gtsam.utils import plot
import utils.plot as plot_helper
from models import Constants
import exercise_4, exercise_6, exercise_3, exercise_5
from exercise_4 import track_camera_for_many_images
from exercise_6 import *
PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker"
PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS = "../../models/bundle_adjustment_results"
K, M1, M2 = utils.read_cameras()
GTSAM_K = utils.get_gtsam_calib_mat(K, M2)

from collections import defaultdict
import heapq
import numpy as np
from collections import defaultdict
from heapq import heappop, heappush
import dijkstar


##################################################################################


def update_pose_graph_with_factor(pose_graph, c0, c1, relative_pose, noise_cov):
    # create a pose factor according to the given parameters
    factor = gtsam.BetweenFactorPose3(c0, c1, relative_pose, noise_cov)
    # add it to the pose graph
    pose_graph.add(factor)
    return pose_graph


def edge_cost_func(cond_mat=None, equal=False):
    if equal:
        return 1
    else:
        return 0.000000001 * np.sqrt(1 / np.linalg.det(10 * cond_mat))


def init_graph_for_shortest_path(pose_graph, key_frames, cond_matrices):
    graph_for_shortest_path = dijkstar.Graph()
    edge_to_covariance = {}
    for i in range(len(key_frames) - 1):
        first = key_frames[i]
        second = key_frames[i + 1]
        c0 = gtsam.symbol(Constants.CAMERA, first)
        c1 = gtsam.symbol(Constants.CAMERA, second)
        edge_cost = edge_cost_func(cond_matrices[i])
        print(edge_cost)
        graph_for_shortest_path.add_edge(first, second, edge_cost)
        edge_to_covariance[(first, second)] = cond_matrices[i]
    return graph_for_shortest_path, edge_to_covariance


def estimate_covariance_by_path(path, edge_to_covariance):
    estimated_cov = np.zeros((6, 6))
    for i in range(len(path) - 1):
        first = path[i]
        second = path[i + 1]
        estimated_cov += edge_to_covariance[(first, second)]
    return estimated_cov


def compute_mahalanobis_distance_between_frames(relative_pose, relative_covariance):
    values = gtsam.Values()
    d0 = gtsam.symbol("d", 0)
    values.insert(d0, relative_pose)
    noise_model = gtsam.noiseModel.Gaussian.Covariance(relative_covariance)
    factor = gtsam.PriorFactorPose3(d0, gtsam.Pose3(), noise_model)
    mahalanobis_distance = factor.error(values)
    return mahalanobis_distance


def find_4_images_matches(consecutive_matches, prev_indices_mapping, cur_indices_mapping, ind_to_3d_point_prev, matcher, reference_kf, candidate_fk):
    """

    :param consecutive_matches:
    :param prev_indices_mapping:
    :param cur_indices_mapping:
    :return:
    """
    try:
        concensus_matces = []
        filtered_matches = []
        tracks = []
        for idx, matches_list in enumerate(consecutive_matches):
            m = matches_list[0]
            prev_left_kp = m.queryIdx
            cur_left_kp = m.trainIdx
            if prev_left_kp in prev_indices_mapping and cur_left_kp in cur_indices_mapping:

                xl, yl = matcher.get_feature_location_frame(reference_kf, kp=prev_left_kp, loc=LEFT)
                xr, yr = matcher.get_feature_location_frame(reference_kf, kp=prev_indices_mapping[prev_left_kp], loc=RIGHT)

                xl_n,yl_n = matcher.get_feature_location_frame(candidate_fk, kp=cur_left_kp, loc=LEFT)
                xr_n,yr_n = matcher.get_feature_location_frame(candidate_fk, kp=cur_indices_mapping[cur_left_kp], loc=RIGHT)

                feature_location_prev = (xl, xr, yl)
                feature_location_cur = (xl_n, xr_n, yl_n)
                tracks.append([feature_location_prev, feature_location_cur])

                prev_left_ind = prev_left_kp
                prev_right_ind = prev_indices_mapping[prev_left_kp]
                cur_left_ind = cur_left_kp
                cur_right_ind = cur_indices_mapping[cur_left_kp]
                point_3d_prev = ind_to_3d_point_prev[prev_left_ind]
                concensus_matces.append((prev_left_ind, prev_right_ind, cur_left_ind, cur_right_ind, point_3d_prev))
                filtered_matches.append(matches_list)

        return concensus_matces, tracks
    except KeyError as e:
        raise e


def consensus_matching_of_general_two_frames(reference_kf, candidate_kf, matcher, thresh=0.6):
    matcher.read_images(reference_kf)
    prev_inlier_indices_mapping = exercise_4.match_next_pair(reference_kf, matcher)
    prev_points_cloud, prev_ind_to_3d_point_dict = exercise_4.get_3d_points_cloud(prev_inlier_indices_mapping, K,
                                                                       M1, M2, matcher,
                                                                       file_index=reference_kf, debug=False)
    prev_indices_mapping = array_to_dict(prev_inlier_indices_mapping)

    cur_inlier_indices_mapping = exercise_4.match_next_pair(candidate_kf, matcher)
    cur_indices_mapping = array_to_dict(cur_inlier_indices_mapping)


    consecutive_matches = matcher.match_between_any_frames(reference_kf, candidate_kf, thresh=thresh)

    consensus_matches, tracks = find_4_images_matches(consecutive_matches, prev_indices_mapping, cur_indices_mapping,
                                                      prev_ind_to_3d_point_dict, matcher, reference_kf, candidate_kf)

    if len(consensus_matches) < 10:
        return None, None, None
    kp1, kp2 = matcher.get_kp(idx=candidate_kf)
    Rt, inliers_ratio, supporter_indices = exercise_4.ransac_for_pnp(consensus_matches, K, kp1, kp2, M2, thresh=2,
                                       debug=False, max_iterations=1000, return_supporters=True)

    if Rt is None:
        return None, None, None
    filtered_tracks = [track for i, track in enumerate(tracks) if supporter_indices[i]]

    return Rt, filtered_tracks, inliers_ratio


def create_trivial_factor_graph(reference_kf, candidate_kf, Rt, filtered_tracks):
    landmarks = set()
    factor_graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    # prior factor + first camera location
    c0 = gtsam.symbol(CAMERA, reference_kf)
    c0_cam_pose = gtsam.Pose3()
    initial_estimates.insert(c0, c0_cam_pose)
    s = 0.1 * np.array([(3 * np.pi / 180), (3 * np.pi / 180), (3 * np.pi / 180)] + [.1, 0.01, 1.0])
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(s)
    prior_factor = gtsam.PriorFactorPose3(c0, c0_cam_pose, prior_noise)
    factor_graph.add(prior_factor)

    # second camera initial location
    c1 = gtsam.symbol(CAMERA, candidate_kf)
    c1_cam_pose = gtsam.Pose3(exercise_5.invert_Rt_transformation(Rt))  # now pose should be from candidate to reference
    initial_estimates.insert(c1, c1_cam_pose)

    # 3d points locations + projection factors
    for i, track in enumerate(filtered_tracks):
        # triangulating the track point and adding it to initial_estimates
        last_frame_camera = gtsam.StereoCamera(c1_cam_pose, GTSAM_K)
        last_loc = track[-1]
        last_point2 = gtsam.StereoPoint2(last_loc[0], last_loc[1], last_loc[2])
        last_point3 = last_frame_camera.backproject(last_point2)
        point_symbol = gtsam.symbol(POINT, i)
        initial_estimates.insert(point_symbol, last_point3)
        landmarks.add(point_symbol)

        # creating projection factor on c0
        first_loc = track[0]
        measured_point2 = gtsam.StereoPoint2(first_loc[0], first_loc[1], first_loc[2])
        stereomodel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        factor = gtsam.GenericStereoFactor3D(measured_point2, stereomodel_noise, c0, point_symbol, GTSAM_K)
        factor_graph.add(factor)

    return factor_graph, initial_estimates, landmarks


def solve_trivial_bundle(reference_kf, candidate_kf, matching_data):
    _, _, Rt, filtered_tracks, inliers_ratio = matching_data
    factor_graph, initial_estimates, landmarks = create_trivial_factor_graph(reference_kf, candidate_kf, Rt, filtered_tracks)
    optimizer = gtsam.LevenbergMarquardtOptimizer(factor_graph, initial_estimates)
    try:
        optimized_estimates = optimizer.optimize()
    except RuntimeError as e:
        print(e)
        return None
    key_vectors = gtsam.KeyVector()
    key_vectors.append(gtsam.symbol(CAMERA, candidate_kf))
    key_vectors.append(gtsam.symbol(CAMERA, reference_kf))  # TODO maybe its the other way around...
    bundle_joint_covariance = gtsam.Marginals(factor_graph, optimized_estimates)
    information_mat_cick = bundle_joint_covariance.jointMarginalInformation(key_vectors).fullMatrix()
    conditional_covariance = np.linalg.inv(information_mat_cick[6:, 6:])
    relative_pose = get_relative_pose_between_frames(candidate_kf, reference_kf, optimized_estimates) # I think we want reference relative to candidate
    return relative_pose, conditional_covariance, factor_graph, optimized_estimates


def loop_closure(pose_graph, key_frames, cond_matrices, pose_graph_initial_estimates, matcher,
                 mahalanobis_thresh=MAHALANOBIS_THRESH, max_candidates_num=5, min_diff_between_loop_frames=5, req_inliers_ratio = 0.9):
    graph_for_shortest_path, edge_to_covariance = \
        init_graph_for_shortest_path(pose_graph=pose_graph, key_frames=key_frames, cond_matrices=cond_matrices)
    cur_pose_graph_estimates = pose_graph_initial_estimates
    # we loop over all key frames, and find possible loop closures for them
    min_md, min_md_index = np.inf, None
    good_ms = []
    successful_lc = []
    for i in range(1, len(key_frames)):
        if i % 20 == 0:
            print(f"-------------- Starting Loop Closing the {i}th Key Frame -----------------")
        candidates_with_small_m_distance = []
        reference_kf = key_frames[i]
        # Step 1: detect loop closure candidates according to graph geometry
        for j in range(i):
            if abs(i - j) <= min_diff_between_loop_frames:
                continue
            prev_kf = key_frames[j]
            shortest_path_info = dijkstar.find_path(graph_for_shortest_path, prev_kf, reference_kf)
            shortest_path_nodes = shortest_path_info.nodes
            estimated_covariance = estimate_covariance_by_path(shortest_path_nodes, edge_to_covariance)
            relative_pose_between_frames = get_relative_pose_between_frames(reference_kf, prev_kf, cur_pose_graph_estimates) # TODO maybe opposite
            mahalanobis_distance = \
                compute_mahalanobis_distance_between_frames(relative_pose_between_frames, estimated_covariance)
            if mahalanobis_distance < min_md:
                min_md = mahalanobis_distance
                min_md_index = i
                print(f"current minimal md found is {min_md} between cur_kf {reference_kf} and prev_kf {prev_kf} ({i}-{j})")
            if mahalanobis_distance < mahalanobis_thresh:
                candidates_with_small_m_distance.append(j)
                good_ms.append((mahalanobis_distance, i, j))
        candidates_with_small_m_distance.sort()
        best_candidates = candidates_with_small_m_distance[:max_candidates_num]

        # Step 2: perform consensus matching between current frame and relevant candidates
        candidates_after_consensus_matching = []
        should_optimize = False
        for candidate in best_candidates:
            print(f"for the {i}th KeyFrame, found {len(best_candidates)} candidates to perform visual odometry with.")
            candidate_kf = key_frames[candidate]
            Rt, filtered_tracks, inliers_ratio = consensus_matching_of_general_two_frames(reference_kf, candidate_kf, matcher)
            if Rt is None:
                print(f"didn't find good transformation between {i}th kf and {candidate}th kf")
                continue
            print(f"number of tracks between {i}th kf and {candidate}th kf after consensus matching is {len(filtered_tracks)}, with inliers ratio of {inliers_ratio}")
            a = 5
            if inliers_ratio > req_inliers_ratio:
                # Step 3: We now optimize this guess by applying a small bundle on it
                print(f"Solving small bundle for {i}-{candidate} with inliers ratio {inliers_ratio}")
                matching_data = (reference_kf, candidate_kf, Rt, filtered_tracks, inliers_ratio)
                candidates_after_consensus_matching.append(matching_data)
                res = solve_trivial_bundle(reference_kf, candidate_kf, matching_data)
                if res is not None:
                    relative_pose, conditional_covariance, small_graph, small_graph_estimates = res
                else:
                    print("Couldn't solve this trivial bundle for some reason... continuing.")
                    continue
                print(f"Small bundle error is {small_graph.error(small_graph_estimates)}")

                # Step 4: we update the pose graph accordingly
                c0 = gtsam.symbol(CAMERA, candidate_kf)
                c1 = gtsam.symbol(CAMERA, reference_kf)
                noise_cov = gtsam.noiseModel.Gaussian.Covariance(conditional_covariance)
                pose_factor = gtsam.BetweenFactorPose3(c0, c1, relative_pose, noise_cov)
                pose_graph.add(pose_factor)
                edge_cost = edge_cost_func(conditional_covariance)
                graph_for_shortest_path.add_edge(candidate, reference_kf, edge_cost)
                edge_to_covariance[(candidate, reference_kf)] = conditional_covariance
                successful_lc.append((i, candidate))
                print(f"Successfully appended kfs ({i}-{candidate}) as a loop closure!")
                should_optimize = True

        if should_optimize:
            optimizer = gtsam.LevenbergMarquardtOptimizer(pose_graph, cur_pose_graph_estimates)
            cur_pose_graph_estimates = optimizer.optimize()
            should_optimize = False
    return pose_graph, cur_pose_graph_estimates, successful_lc







def plot_loop_between_two_frames(our_trajectory, first, second, key_frames, path="lc_"):
    camera_positions = np.array([pose.translation() for pose in our_trajectory])
    plt.figure()
    plt.scatter(x=camera_positions[:, 0], y=camera_positions[:, 2], color='blue', label='our trajectory', s=0.75)
    plt.scatter(x=camera_positions[first, 0], y=camera_positions[first, 2], color='green', label='first frame', s=20)
    plt.scatter(x=camera_positions[second, 0], y=camera_positions[second, 2], color='red', label='second frame', s=20)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(f"compare frame {key_frames[first]} with {key_frames[second]}")
    plt.legend()
    plt.savefig(path + f"trajectory_{first}_{second}")
    draw_matching_images(first, second, key_frames)


def draw_matching_images(first, second, key_frames):
    file_index1, file_index2 = key_frames[first], key_frames[second]
    left1, _ = read_images(file_index1)
    left2, _ = read_images(file_index2)
    img3 = cv2.vconcat([cv2.cvtColor(left1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(left2, cv2.COLOR_GRAY2BGR)])
    plt.figure(figsize=(16, 9))
    plt.imshow(img3)
    plt.title(f"compare image {file_index1} and {file_index2}")
    plt.savefig(f"lc_compare_images_{first}_{second}")


if __name__ == "__main__":
    matcher = Matcher()

    random.seed(6)
    s = gtsam.StereoCamera()
    track_db = TrackDatabase()
    deserialization_result = track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
    if deserialization_result == FAILURE:
        _, track_db = exercise_4.track_camera_for_many_images()
        track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)
    bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
                                    cond_matrices = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
    key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
    pose_graph, initial_estimates, landmarks = exercise_6.create_pose_graph(bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, cond_matrices)
    kf_to_covariance = {key_frames[i + 1]: cond_matrices[i] for i in range(len(cond_matrices))}
    cond_matrices = [cond_matrix * 10 for cond_matrix in cond_matrices]
    our_trajectory = optimized_global_keyframes_poses
    # matcher_cache = track_db.get_matcher_cache()
    # matcher.set_cache(matcher_cache)
    pose_graph, cur_pose_graph_estimates, successful_lc = loop_closure(pose_graph, key_frames,
                                                                       matcher=matcher, cond_matrices=cond_matrices,
                                                                       mahalanobis_thresh=MAHALANOBIS_THRESH,
                                                                       pose_graph_initial_estimates=initial_estimates)
