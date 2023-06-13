import random
import cv2
import numpy as np
from tqdm import tqdm

from VAN_ex.code.exercise_2 import least_squares
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from utils import utils
from utils.plotters import draw_3d_points, draw_inlier_and_outlier_matches, draw_matches, plot_four_cameras, \
    draw_supporting_matches, plot_trajectories, plot_regions_around_matching_pixels, plot_dict, plot_connectivity_graph, \
    gen_hist, plot_reprojection_errors
from utils.utils import *
from matplotlib import pyplot as plt
from models.Constants import *
import gtsam
from gtsam.utils import plot

PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker"
K, M1, M2 = utils.read_cameras()
GTSAM_K = utils.get_gtsam_calib_mat(K, M2)

compose = lambda first_matrix, last_matrix: last_matrix @ np.append(first_matrix, [np.array([0, 0, 0, 1])], axis=0)

def criteria(frameIds, percentage=.82):
    """
    choosing proper keyframes using a median criterion
    """
    # Sort frames based on track length (ascending order)
    sorted_frames = sorted(frameIds, key=lambda frameId: len(track_db.get_track_ids_for_frame(frameId)))
    for fid in sorted_frames:
        l = len(track_db.get_track_ids_for_frame(fid))
        print(f"frame id {fid} has {l} tracks")
    # Calculate the index of the median frame
    median_index = int(len(sorted_frames) * percentage)

    # Select the keyframes from the median frame to the last frame
    key_frames = sorted_frames[median_index:]
    # return [frameId for frameId in range(1, len(frameIds),5)]
    return key_frames

def get_bundle_windows(key_frames):
    return [(key_frames[i - 1], key_frames[i]) for i in range(1, len(key_frames))]


def compute_distance(pose_i, pose_j):
    translation_i = pose_i[:3, 3]  # Extract translation vector from pose_i
    translation_j = pose_j[:3, 3]  # Extract translation vector from pose_j

    distance = np.linalg.norm(translation_j - translation_i)
    return distance


def select_keyframes_distance(frames, min_distance_threshold=1, max_keyframes=20):
    distances = []
    for i in range(len(frames) - 1):
        frameId = frames[i]
        next_frameId = frames[i+1]
        pose_i = track_db.get_extrinsic_matrix_by_frameId(frameId)
        pose_j = track_db.get_extrinsic_matrix_by_frameId(next_frameId)
        distance = compute_distance(pose_i, pose_j)
        distances.append(distance)

    # Find frames with distances above the threshold
    keyframe_indices = [0]  # Start with the first frame as a keyframe
    for i, distance in enumerate(distances):
        if distance > min_distance_threshold:
            keyframe_indices.append(i + 1)  # Add 1 to account for indexing difference

    # Limit the number of keyframes
    if len(keyframe_indices) > max_keyframes:
        keyframe_indices = keyframe_indices[:max_keyframes]

    return [frames[idx] for idx in keyframe_indices]


def create_factor_graph(bundle_window_frameIds):
    """
    Creates the factor graph for the bundle window
    """
    cam_pose = None
    landmarks = set()
    # bundle_window_frameIds = [0, 4]
    # Compute the first frame's extrinsic matrix that maps points from camera coordinates to world coordinates
    first_cam_pose = track_db.get_extrinsic_matrix_by_frameId(bundle_window_frameIds[0])
    bundle_starts_in_frame_id = bundle_window_frameIds[0]
    bundle_ends_in_frame_id = bundle_window_frameIds[-1]
    # Initialize the factor graph and values
    factor_graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    # Create factors and values for each frame
    for i, frameId in enumerate(range(bundle_starts_in_frame_id, bundle_ends_in_frame_id)):
        # Create camera symbol and update values dictionary
        cam_pose_sym = gtsam.symbol(CAMERA, frameId)
        cur_cam_pose = extinsic_to_global(compose(first_cam_pose, track_db.get_extrinsic_matrix_by_frameId(frameId)))
        cam_pose = gtsam.Pose3(cur_cam_pose)
        initial_estimates.insert(cam_pose_sym, gtsam.Pose3(cam_pose))

        if i == 0:
            # Add prior factor for the first frame
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas([1,1,1,1,1,1])
            factor_graph.add(gtsam.PriorFactorPose3(cam_pose_sym, cam_pose, prior_noise))

    # Get all the tracks that related to this specific bundle window
    relevant_tracks = get_only_relevant_tracks(bundle_ends_in_frame_id, bundle_starts_in_frame_id)
    # For each relevant track, create measurement factors
    for track_data, trackId in relevant_tracks:
        # track_data = track_db.get_track_data(track_id)
        track_ends_in_frame_id = track_data[-1][-1]
        if track_ends_in_frame_id < bundle_ends_in_frame_id:
            continue

        # Create measurement factor for this track point
        last_frame_pose = gtsam.StereoCamera(cam_pose, GTSAM_K)
        last_loc = track_data[-1][LOCATIONS_IDX]
        # Get the 2D point in the last frame for triangulation
        last_point2 = gtsam.StereoPoint2(last_loc[0], last_loc[1], last_loc[2])
        last_point3 = last_frame_pose.backproject(last_point2)

        point_symbol = gtsam.symbol(POINT, trackId)
        initial_estimates.insert(point_symbol, last_point3)
        landmarks.add(point_symbol)
        locations = np.array(track_data, dtype=object)[:,1,...]
        for i, frame_id in enumerate(range(bundle_starts_in_frame_id, bundle_ends_in_frame_id)):
            cam_symbol = gtsam.symbol(CAMERA, frame_id)
            # Get the measured 2D point in the current frame
            measured_point2 = gtsam.StereoPoint2(locations[i][0], locations[i][1], locations[i][2])

            # Create the factor between the measured and projected points
            stereomodel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2, 2, 2]))

            factor = gtsam.GenericStereoFactor3D(measured_point2, stereomodel_noise, cam_symbol, point_symbol, GTSAM_K)

            # Add the factor to the factors list
            factor_graph.add(factor)

        # create_measurement_factor(track_id, track_point)
    return factor_graph, initial_estimates, landmarks


def get_only_relevant_tracks(bundle_ends_in_frame_id, bundle_starts_in_frame_id):
    tracksIds = track_db.get_track_ids_for_frame(bundle_starts_in_frame_id)
    tracks = [(track_db.get_track_data(trackId), trackId) for trackId in tracksIds]
    relevant_tracks = [(track, trackId) for (track, trackId) in tracks if track[-1][-1] >= bundle_ends_in_frame_id]
    return relevant_tracks


def triangulate_and_project(track):
    """
    Triangulate a 3D point from the last frame of the track
    and project it to all frames in the track.
    Return the reprojection errors and factors.
    """

    frameIds = [track_feature[FRAME_ID] for track_feature in track]
    locations = [track_feature[LOCATIONS_IDX] for track_feature in track]

    right_points = [(xr, y) for _, xr, y in locations]
    left_points = [(xl, y) for xl, _, y in locations]

    # Triangulation from the last frame
    x_r_last, y_last = right_points[-1]
    x_l_last, y_last = left_points[-1]

    reprojection_errors = []
    factors = []
    values = gtsam.Values()

    # Triangulate 3D point from the last frame
    last_camera_matrix = track_db.get_extrinsic_matrix_by_frameId(frameIds[-1])
    first_frame_camera_matrix = extinsic_to_global(track_db.get_extrinsic_matrix_by_frameId(frameIds[0]))
    last_camera = compose(first_frame_camera_matrix, last_camera_matrix)
    last_camera_global = extinsic_to_global(last_camera)

    last_cam_pose = gtsam.Pose3(last_camera_global)
    last_frame_pose = gtsam.StereoCamera(last_cam_pose, GTSAM_K)
    last_point2 = gtsam.StereoPoint2(x_l_last, x_r_last, y_last)  # Get the 2D point in the last frame for triangulation
    last_point3 = last_frame_pose.backproject(last_point2)

    # Add the 3D point to the values dictionary
    point_symbol = gtsam.symbol(POINT, 0)
    values.insert(point_symbol, last_point3)
    # current_transformations_from_first_frame = lambda mat: compose(first_frame_camera_matrix, mat)
    for i, frame_id in enumerate(frameIds):
        frame_pose = get_pose_for_frameId(first_frame_camera_matrix, frame_id)
        # Insert left pose symbol for the current frameId:
        cam_symbol = gtsam.symbol(CAMERA, frame_id)
        values.insert(cam_symbol, frame_pose)

        # Get the measured 2D point in the current frame
        measured_point2 = gtsam.StereoPoint2(locations[i][0], locations[i][1], locations[i][2])

        # Project the 3D point to the current frame
        frame_pose = gtsam.StereoCamera(frame_pose, GTSAM_K)
        projected_point2 = frame_pose.project(last_point3)

        # Create the factor between the measured and projected points
        stereomodel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2.0, 1.0]))
        factor = gtsam.GenericStereoFactor3D(measured_point2, stereomodel_noise, cam_symbol, point_symbol, GTSAM_K)

        # Add the factor to the factors list
        factors.append(factor)

        # Compute and store the reprojection error
        measured_point2 = measured_point2.uL(), measured_point2.uR(), measured_point2.v()
        projected_point2 = projected_point2.uL(), projected_point2.uR(), projected_point2.v()
        reprojection_errors.append(np.linalg.norm(np.array(measured_point2) - np.array(projected_point2)))

    return reprojection_errors, factors, values


def get_pose_for_frameId(first_frame_camera_matrix, frame_id):
    """
    Get the pose for a specific frame ID relative to the first frame.

    Args:
        first_frame_camera_matrix (numpy.ndarray): Extrinsic camera matrix of the first frame.
        frame_id (int): Frame ID for which to get the pose.
        track_db (TrackDatabase): TrackDatabase object containing the necessary data.

    Returns:
        gtsam.Pose3: Pose of the frame relative to the first frame.
    """
    # Get the extrinsic matrix for the frame
    ex_mat = extinsic_to_global(track_db.get_extrinsic_matrix_by_frameId(frame_id))

    # Compute the camera pose relative to the first frame
    camera_wrt_first_frame = compose(first_frame_camera_matrix, ex_mat)
    current_pose = extinsic_to_global(camera_wrt_first_frame)

    # Convert the pose to a gtsam.Pose3 object
    frame_pose = gtsam.Pose3(current_pose)

    return frame_pose


def get_pose_symbol(key_frame):
    """
    Get the symbol for a camera pose.

    Args:
        key_frame (int): Key frame ID.

    Returns:
        gtsam.symbol: Symbol for the camera pose.
    """
    return gtsam.symbol(CAMERA, key_frame)


def get_p3d_symbol(frame_id):
    """
    Get the symbol for a 3D point landmark.

    Args:
        p3d (int): 3D point ID.

    Returns:
        gtsam.symbol: Symbol for the 3D point landmark.
    """
    return gtsam.symbol(POINT, frame_id)

def compute_factor_error(factor, values):
    """
    Compute the factor error given a factor and a values dictionary.
    """
    error = factor.unwhitenedError(values)
    return np.linalg.norm(error)


def q1():
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    trackId, track = track_db.get_random_track_of_length(10)
    reprojection_errors, factors, values = triangulate_and_project(track)
    # Plot the reprojection errors
    print(min(reprojection_errors), max(reprojection_errors))
    print(f"f(factor_err) ")

    plt.plot(range(len(reprojection_errors)), reprojection_errors)
    plt.xlabel('Frame Index')
    plt.ylabel('Reprojection Error')
    plt.title('Reprojection Error over Track')
    plt.show()

    # Plot the factor errors
    factor_errors = [compute_factor_error(factor, values) for factor in factors]
    plt.plot(range(track[-1][-1]-track[0][-1]+1), factor_errors)
    plt.xlabel('Frame Index')
    plt.ylabel('Factor Error')
    plt.title('Factor Error over Track')
    plt.show()


def q2():
    # Step 1: Select Keyframes
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    # key_frames1 = criteria1(frameIds=frameIds)
    key_frames = criteria(frameIds)
    # select_keyframes_distance(frameIds.keys())
    bundle_windows = get_bundle_windows(key_frames)
    for bw in bundle_windows:
        print(f"{bw} window size is {bw[1]- bw[0]}")
    # Step 2: Define Bundle Optimization
    first_bundle = bundle_windows[0]
    bundle_starts_in_frame_id = first_bundle[0]
    bundle_ends_in_frame_id = first_bundle[-1]
    # Step 3: Add Factors and Initial Estimates for Keyframes and Landmarks
    bundle_graph, initial_estimates, landmarks = create_factor_graph(first_bundle)

    # Step 4: Perform Bundle Adjustment Optimization
    optimizer = gtsam.LevenbergMarquardtOptimizer(bundle_graph, initial_estimates)
    optimized_estimates = optimizer.optimize()

    # Step 5: Print Total Factor Graph Error
    initial_error = bundle_graph.error(initial_estimates)
    optimized_error = bundle_graph.error(optimized_estimates)

    print("Initial Total Factor Graph Error:", initial_error)
    print("Optimized Total Factor Graph Error:", optimized_error)

    # Step 6: Plot the Resulting Positions
    # Plotting the trajectory as a 3D graph
    # gtsam.utils.plot.set_axes_equal(0)
    gtsam.utils.plot.plot_trajectory(fignum=0, values=optimized_estimates, title="Bundle Adjustment Trajectory")
    plt.show()
    optimized_landmarks = [optimized_estimates.atPoint3(lm_sym) for lm_sym in landmarks]
    # Step 6: Pick a Projection Factor and Compute Error
    frame_c = key_frames[0]

    initial_error_projection = bundle_graph.error(initial_estimates)
    optimized_error_projection = bundle_graph.error(optimized_estimates)

    print("Initial Projection Factor Error:", initial_error_projection)
    print("Optimized Projection Factor Error:", optimized_error_projection)



if __name__ == "__main__":
    random.seed(6)
    s = gtsam.StereoCamera()
    track_db = TrackDatabase()
    deserialization_result = track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
    if deserialization_result == FAILURE:
        _, track_db = track_camera_for_many_images()
        track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)

    q1()

    q2()
