import random
import cv2
import numpy as np
from tqdm import tqdm
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
    cam_matrix = gtsam.Pose3(initial_estimates.atPose3(symbol(CAMERA_SYM, key_frame)))
    top_candidates = []
    for candidate_frame_id in closure_candidates:
        min_cov_path = graph.sum_cov_along_path()
        candidate_cam_matrix = gtsam.Pose3(initial_estimates.atPose3(symbol(CAMERA_SYM, candidate_frame_id)))
        diff_between_cams = candidate_cam_matrix.between(cam_matrix)


        # Perform Mahalanobis distance thresholding
        # TODO: Retrieve the covariance
        if get_mahalanobis_distance(diff_between_cams, covariance=None) < mahalanobis_thresh:
            top_candidates.append(candidate_frame_id)

    return top_candidates


def loop_closure(key_frames, mahalanobis_thresh=MAHALANOBIS_THRESH):
    graph = Graph(bundle_results, cond_matrices)
    candidates = find_all_closures_candidates(key_frames) # key frame, closure_candidates = candidates[i]
    for key_frame, closure_candidates in candidates:
        check_for_closure(key_frame, closure_candidates,mahalanobis_thresh)


def get_mahalanobis_distance(diff_between_cams, covariance):
    # Convert diff_between_cams pose to a 6-dimensional vector
    diff_vector = gtsam.Pose3.Log(diff_between_cams).vector()
    return np.sqrt(diff_vector.T @ inv(covariance) @ diff_vector)


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

    pose_graph, initial_estimates, landmarks = exercise_6.create_pose_graph(bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, cond_matrices)



    matcher_cache = track_db.get_matcher_cache()
    matcher.set_cache(matcher_cache)

    pass