import cv2
import dijkstar
from matplotlib import pyplot as plt

import models.BundleAdjustment
from utils.utils import read_cameras, get_gtsam_calib_mat, get_3d_points_cloud
from utils import plot
from models import Constants
from models.BundleAdjustment import *
from models.Constants import *
from utils.plotters import plot_uncertainty_over_time, plot_trajectories, plot_localization_error_over_time
from utils.utils import array_to_dict, read_images, get_gt_trajectory
K, M1, M2 = read_cameras()
GTSAM_K = get_gtsam_calib_mat(K, M2)

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
    costs = []
    for i in range(len(key_frames) - 1):
        first = key_frames[i]
        second = key_frames[i + 1]
        c0 = gtsam.symbol(Constants.CAMERA, first)
        c1 = gtsam.symbol(Constants.CAMERA, second)
        edge_cost = edge_cost_func(cond_matrices[i])
        costs.append(edge_cost)
        # print(edge_cost)
        graph_for_shortest_path.add_edge(first, second, edge_cost)
        edge_to_covariance[(first, second)] = cond_matrices[i]
    costs = sorted(costs)

    print(costs[len(costs)//10])
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

        return concensus_matces, tracks, filtered_matches
    except KeyError as e:
        raise e


def consensus_matching_of_general_two_frames(reference_kf, candidate_kf, matcher, thresh=0.6):
    matcher.read_images(reference_kf)
    prev_inlier_indices_mapping = utils.utils.match_next_pair(reference_kf, matcher)
    prev_points_cloud, prev_ind_to_3d_point_dict = get_3d_points_cloud(prev_inlier_indices_mapping, K,
                                                                       M1, M2, matcher,
                                                                       file_index=reference_kf, debug=False)
    prev_indices_mapping = array_to_dict(prev_inlier_indices_mapping)

    cur_inlier_indices_mapping = utils.utils.match_next_pair(candidate_kf, matcher)
    cur_indices_mapping = array_to_dict(cur_inlier_indices_mapping)


    consecutive_matches = matcher.match_between_any_frames(reference_kf, candidate_kf, thresh=thresh)

    consensus_matches, tracks, filtered_matches = find_4_images_matches(consecutive_matches, prev_indices_mapping, cur_indices_mapping,
                                                      prev_ind_to_3d_point_dict, matcher, reference_kf, candidate_kf)

    if len(consensus_matches) < 10:
        return None, None, None, None, None
    kp1, kp2 = matcher.get_kp(idx=candidate_kf)
    Rt, inliers_ratio, supporter_indices = utils.utils.ransac_for_pnp(consensus_matches, K, kp1, kp2, M2, thresh=2,
                                                                      debug=False, max_iterations=500, return_supporters=True)

    if Rt is None:
        return None, None, None, None, None
    filtered_tracks = [track for i, track in enumerate(tracks) if supporter_indices[i]]

    return Rt, filtered_tracks, inliers_ratio, filtered_matches, supporter_indices


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
    c1_cam_pose = gtsam.Pose3(invert_Rt_transformation(Rt))  # now pose should be from candidate to reference
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

        # creating projection factor on c1
        measured_point2 = gtsam.StereoPoint2(last_loc[0], last_loc[1], last_loc[2])
        stereomodel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        factor = gtsam.GenericStereoFactor3D(measured_point2, stereomodel_noise, c1, point_symbol, GTSAM_K)
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
                 mahalanobis_thresh=MAHALANOBIS_THRESH, max_candidates_num=5, min_diff_between_loop_frames=20,
                 req_inliers_ratio=0.85, draw_supporting_matches_flag=False, points_to_stop_by=False,
                 compare_to_gt=False, show_localization_error=False, show_uncertainty=False):
    graph_for_shortest_path, edge_to_covariance = \
        init_graph_for_shortest_path(pose_graph=pose_graph, key_frames=key_frames, cond_matrices=cond_matrices)
    cur_pose_graph_estimates = pose_graph_initial_estimates
    # we loop over all key frames, and find possible loop closures for them
    min_md, min_md_index = np.inf, None
    good_ms = []
    successful_lc = []
    prev_num_of_successful_lc = 0
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
            Rt, filtered_tracks, inliers_ratio, matches, supporting_indices = consensus_matching_of_general_two_frames(reference_kf, candidate_kf, matcher)
            if Rt is None:
                print(f"didn't find good transformation between {i}th kf and {candidate}th kf")
                continue
            print(f"number of tracks between {i}th kf and {candidate}th kf after consensus matching is {len(filtered_tracks)}, with inliers ratio of {inliers_ratio}")
            if inliers_ratio > req_inliers_ratio:
                if draw_supporting_matches_flag:
                    # draw_supporting_matches_general(candidate_kf, reference_kf, matcher, matches, supporting_indices)
                    draw_supporting_matches_flag = False
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
                noise_cov = gtsam.noiseModel.Gaussian.Covariance(conditional_covariance * 0.01)
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
            if points_to_stop_by and (len(successful_lc) - prev_num_of_successful_lc) > 5:
                marginals = gtsam.Marginals(pose_graph, cur_pose_graph_estimates)
                plot.plot_trajectory(1, cur_pose_graph_estimates, marginals=marginals,
                                     title=f"optimized estimates trajectory with cov after: {len(successful_lc)} updates", scale=1,
                                     save_file=f"plots/optimized_estimates_trajectory_with_cov_after_{len(successful_lc)}")
                prev_num_of_successful_lc = len(successful_lc)


    if compare_to_gt:
        plot_pg_locations_before_and_after_lc(pose_graph, cur_pose_graph_estimates)
    if show_localization_error:
        plot_pg_locations_error_graph_before_and_after_lc(pose_graph, cur_pose_graph_estimates)
    if show_uncertainty:
        plot_pg_uncertainty_before_and_after_lc(pose_graph, cur_pose_graph_estimates)
    print(f"Overall, {len(successful_lc)} loop closures were detected.")

    return pose_graph, cur_pose_graph_estimates, successful_lc


def get_trajectory_from_graph(values):
    all_poses = gtsam.utilities.extractPose3(values).reshape(-1, 4, 3).transpose(0, 2, 1)
    trajectory = all_poses[:, :, -1]
    return trajectory


def plot_pg_uncertainty_before_and_after_lc(pose_graph_after, values_after):
    bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
    cond_matrices = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
    key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
    pose_graph, initial_estimates, landmarks = models.BundleAdjustment.create_pose_graph(bundle_results,
                                                                                         optimized_relative_keyframes_poses,
                                                                                         optimized_global_keyframes_poses,
                                                                                         cond_matrices)
    pose_graph_after_covariance = gtsam.Marginals(pose_graph_after, values_after)
    pose_graph_before_covariance = gtsam.Marginals(pose_graph, initial_estimates)
    after = []
    before = []
    for kf in key_frames:
        key_vectors = gtsam.KeyVector()
        key_vectors.append(gtsam.symbol(CAMERA, kf))
        kf_cov_after = pose_graph_after_covariance.jointMarginalCovariance(key_vectors).fullMatrix()
        kf_cov_before = pose_graph_before_covariance.jointMarginalCovariance(key_vectors).fullMatrix()
        after.append(kf_cov_after)
        before.append(kf_cov_before)
    after_score = [np.abs(np.linalg.det(cov)) for cov in after]
    before_score = [np.abs(np.linalg.det(cov)) for cov in before]

    plot_uncertainty_over_time(key_frames, after_score, "plots/lc_uncertainty_after", "(after)")
    plot_uncertainty_over_time(key_frames, before_score, "plots/lc_uncertainty_before", "(before)")


def plot_pg_locations_before_and_after_lc(pose_graph_after, values_after):
    bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
    cond_matrices = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
    key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
    pose_graph, initial_estimates, landmarks = models.BundleAdjustment.create_pose_graph(bundle_results,
                                                                                         optimized_relative_keyframes_poses,
                                                                                         optimized_global_keyframes_poses,
                                                                                         cond_matrices)
    trajectory_after = get_trajectory_from_graph(values_after)
    trajectory_before = get_trajectory_from_graph(initial_estimates)
    gt_trajectory = get_gt_trajectory()[key_frames]
    plot_trajectories(trajectory_before, gt_trajectory, path="plots/lc_vs_gt_before", suffix="before_lc")
    plot_trajectories(trajectory_after, gt_trajectory, path="plots/lc_vs_gt_after", suffix="after_lc")


def plot_pg_locations_error_graph_before_and_after_lc(pose_graph_after, values_after):
    bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
    cond_matrices = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
    key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
    pose_graph, initial_estimates, landmarks = models.BundleAdjustment.create_pose_graph(bundle_results,
                                                                                         optimized_relative_keyframes_poses,
                                                                                         optimized_global_keyframes_poses,
                                                                                         cond_matrices)
    trajectory_after = get_trajectory_from_graph(values_after)
    trajectory_before = get_trajectory_from_graph(initial_estimates)
    gt_trajectory = get_gt_trajectory()[key_frames]

    plot_localization_error_over_time(key_frames, trajectory_after, gt_trajectory, path=PATH_TO_SAVE_LOCALIZATION_ERROR_LOOP_CLOSURE_AFTER)
    plot_localization_error_over_time(key_frames, trajectory_before, gt_trajectory, path=PATH_TO_SAVE_LOCALIZATION_ERROR_LOOP_CLOSURE_BEFORE)



