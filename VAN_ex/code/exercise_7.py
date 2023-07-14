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
import exercise_4, exercise_6
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


class GraphNode:
    def __init__(self, index, weight=None):
        self.index = index
        self.weight = weight

class GraphEdge:
    def __init__(self, source, target, covariance=None):
        self.source = source
        self.target = target
        self.covariance = covariance
        self.weight = self.calculate_weight(covariance)

    def calculate_weight(self, covariance):
        if covariance is None:
            return 0
        return np.sqrt(np.linalg.det(covariance))

class Graph:
    def __init__(self, bundle_results, covariance_list):

        self.num_vertices = len(bundle_results[BUNDLE_WINDOW_INDEX])
        self.edges = defaultdict(list)
        # if covariance_list is not None:
        self.construct_graph(covariance_list)

    def construct_graph(self, covariance_list):
        for i in range(len(covariance_list)):
            self.add_edge(i, i + 1, covariance_list[i])

    def add_edge(self, source, target, covariance):
        edge = GraphEdge(source, target, covariance)
        self.edges[source].append(edge)

    def get_shortest_path(self, source, target):
        distances = [np.inf] * self.num_vertices
        distances[source] = 0
        parents = [-1] * self.num_vertices
        queue = [(0, source)]

        while queue:
            _, current_node = heappop(queue)
            if current_node == target:
                break
            for edge in self.edges[current_node]:
                if distances[edge.target] > distances[current_node] + edge.weight:
                    distances[edge.target] = distances[current_node] + edge.weight
                    parents[edge.target] = current_node
                    heappush(queue, (distances[edge.target], edge.target))

        path = []
        current_node = target
        while current_node != -1:
            path.append(current_node)
            current_node = parents[current_node]
        return path[::-1]  # Return path from source to target

    def get_shortest_path2(self, source, target):
        dist = [np.zeros((6, 6))] * self.num_vertices
        dist[source] = np.zeros((6, 6))
        parents = [-1] * self.num_vertices
        queue = [(np.zeros((6, 6)), source)]

        while queue:
            _, current_node = heappop(queue)
            if current_node == target:
                break
            for edge in self.edges[current_node]:
                new_dist = dist[current_node] + edge.covariance
                if np.linalg.norm(new_dist) < np.linalg.norm(dist[edge.target]):
                    dist[edge.target] = new_dist
                    parents[edge.target] = current_node
                    heappush(queue, (dist[edge.target], edge.target))

        path = []
        current_node = target
        while current_node != -1:
            path.append(current_node)
            current_node = parents[current_node]
        return path[::-1]  # Return path from source to target

    def estimate_covariance(self, path):
        estimated_cov = np.zeros((6, 6))
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            for edge in self.edges[source]:
                if edge.target == target:
                    estimated_cov += edge.covariance
                    break
        return estimated_cov

    def sum_cov_along_path(self, path):
        total_cov = np.zeros((6, 6))
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            for edge in self.edges[source]:
                if edge.target == target:
                    total_cov += edge.covariance
                    break
        return total_cov



def generalized_consensus_matcher(frame_loop_candidate, track_db, threshold=.1, matching_threshold=1):
    matcher = Matcher()
    matcher_cache = track_db.get_matcher_cache()
    matcher.set_cache(matcher_cache)
    closure_detection_candidates = []
    for some_prev_frame in range(frame_loop_candidate-1):
        matching, ratio = matcher.match_between_any_frames(reference_frame=frame_loop_candidate, other_frame=some_prev_frame,
                                         thresh=matching_threshold)

    # If inliers to total matches' ratio is significantly high
        if ratio > threshold:
            closure_detection_candidates.append(some_prev_frame)

    return closure_detection_candidates


def find_all_closures_candidates(key_frames):
    candidates = {}
    for frame_id in key_frames:
        candidates_i = generalized_consensus_matcher(frame_id, track_db)
        candidates[frame_id] = candidates_i

    return candidates


def check_for_closure(key_frame, closure_candidates, mahalanobis_thresh):
    initial_estimates = bundle_results[INITIAL_ESTIMATES_INDEX]
    cam_matrix = gtsam.Pose3(initial_estimates.atPose3(gtsam.symbol(CAMERA, key_frame)))
    top_candidates = []
    for candidate_frame_id in closure_candidates:
        min_cov_path = 1#graph.sum_cov_along_path()
        candidate_cam_matrix = gtsam.Pose3(initial_estimates.atPose3(gtsam.symbol(CAMERA, candidate_frame_id)))
        diff_between_cams = candidate_cam_matrix.between(cam_matrix)


        # Perform Mahalanobis distance thresholding
        # TODO: Retrieve the covariance
        if get_mahalanobis_distance(diff_between_cams, covariance=None) < mahalanobis_thresh:
            top_candidates.append(candidate_frame_id)

    return top_candidates


def loop_closure1(key_frames, mahalanobis_thresh=MAHALANOBIS_THRESH):
    graph = Graph(bundle_results, cond_matrices)
    candidates = find_all_closures_candidates(key_frames) # key frame, closure_candidates = candidates[i]
    for key_frame, closure_candidates in candidates:
        check_for_closure(key_frame, closure_candidates,mahalanobis_thresh)


def get_mahalanobis_distance(diff_between_cams, covariance):
    # Convert diff_between_cams pose to a 6-dimensional vector
    diff_vector = gtsam.Pose3.Log(diff_between_cams).vector()
    return np.sqrt(diff_vector.T @ inv(covariance) @ diff_vector)


##################################################################################


def update_pose_graph_with_factor(pose_graph, c0, c1, relative_pose, noise_cov):
    # create a pose factor according to the given parameters
    factor = gtsam.BetweenFactorPose3(c0, c1, relative_pose, noise_cov)
    # add it to the pose graph
    pose_graph.add(factor)
    return pose_graph


def init_graph_for_shortest_path(pose_graph, key_frames, cond_matrices):
    graph_for_shortest_path = dijkstar.Graph()
    edge_to_covariance = {}
    for i in range(len(key_frames) - 1):
        first = key_frames[i]
        second = key_frames[i + 1]
        c0 = gtsam.symbol(Constants.CAMERA, first)
        c1 = gtsam.symbol(Constants.CAMERA, second)
        edge_cost = 0.000000001 * np.sqrt(1 / np.linalg.det(10 * cond_matrices[i]))
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


def consensus_matching_of_general_two_frames(reference_kf, candidate_kf, matcher, thresh=0.6):
    matcher.read_images(reference_kf)
    small_track_db = TrackDatabase()
    prev_inlier_indices_mapping = exercise_4.match_next_pair(reference_kf, matcher)
    prev_points_cloud, prev_ind_to_3d_point_dict = exercise_4.get_3d_points_cloud(prev_inlier_indices_mapping, K,
                                                                       M1, M2, matcher,
                                                                       file_index=reference_kf, debug=False)
    prev_indices_mapping = array_to_dict(prev_inlier_indices_mapping)

    cur_inlier_indices_mapping = exercise_4.match_next_pair(candidate_kf, matcher)
    cur_indices_mapping = array_to_dict(cur_inlier_indices_mapping)


    consecutive_matches = matcher.match_between_any_frames(reference_kf, candidate_kf, thresh=thresh)

    consensus_matches, filtered_matches = exercise_4.consensus_match(consecutive_matches, prev_indices_mapping,
                                                          cur_indices_mapping, prev_ind_to_3d_point_dict, track_db,
                                                          frameId, matcher)

    kp1, kp2 = matcher.get_kp(idx=candidate_kf)
    Rt, inliers_ratio = exercise_4.ransac_for_pnp(consensus_matches, K, kp1, kp2, M2, thresh=2,
                                       debug=False, max_iterations=1000)

    # Loop over all frames
    for frameId in tqdm(range(num_of_frames-1)):
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
        Rt, inliers_ratio = ransac_for_pnp(consensus_matches, k, kp1, kp2, m2, thresh=2,
                            debug=False, max_iterations=1000)
    return None, None


def loop_closure(pose_graph, key_frames, cond_matrices, pose_graph_initial_estimates, matcher,
                 mahalanobis_thresh=MAHALANOBIS_THRESH, max_candidates_num=5, min_diff_between_loop_frames=5):
    graph_for_shortest_path, edge_to_covariance = \
        init_graph_for_shortest_path(pose_graph=pose_graph, key_frames=key_frames, cond_matrices=cond_matrices)
    cur_pose_graph_estimates = pose_graph_initial_estimates
    # we loop over all key frames, and find possible loop closures for them
    min_md, min_md_index = np.inf, None
    good_ms = []
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
        for candidate in best_candidates:
            candidate_kf = key_frames[candidate]
            Rt, inliers_ratio = consensus_matching_of_general_two_frames(reference_kf, candidate_kf)
    a = 5






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
    matcher_cache = track_db.get_matcher_cache()
    matcher.set_cache(matcher_cache)
    loop_closure(pose_graph, key_frames, matcher=matcher, cond_matrices=cond_matrices,
                 mahalanobis_thresh=MAHALANOBIS_THRESH, pose_graph_initial_estimates=initial_estimates)




    pass