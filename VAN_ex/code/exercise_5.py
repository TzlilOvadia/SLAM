import random
import cv2
import numpy as np
from tqdm import tqdm

from VAN_ex.code.exercise_2 import least_squares
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from utils import utils
from utils.plotters import draw_3d_points, draw_inlier_and_outlier_matches, draw_matches, plot_four_cameras, \
    draw_supporting_matches, plot_trajectories, plot_regions_around_matching_pixels, plot_dict, plot_connectivity_graph, gen_hist, plot_reprojection_errors
from utils.utils import *
from matplotlib import pyplot as plt
from models.Constants import *
import gtsam

PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker"
K, M1, M2 = utils.read_cameras()
GTSAM_K = utils.get_gtsam_calib_mat(K, M2)

compose = lambda first_matrix, last_matrix: last_matrix @ np.append(first_matrix, [np.array([0, 0, 0, 1])], axis=0)
stereomodel = lambda a, b: gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0]))

def q1():
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    trackId, track = track_db.get_random_track_of_length(10)

    reprojection_errors, factors, values = triangulate_and_project(track)

    # Plot the reprojection errors
    plt.plot(range(len(reprojection_errors)), reprojection_errors)
    plt.xlabel('Frame Index')
    plt.ylabel('Reprojection Error')
    plt.title('Reprojection Error over Track')
    plt.show()



def criteria(frameIds, percentage=.85):
    """
    choosing proper keyframes using a median criterion
    """
    # Sort frames based on track length (ascending order)
    sorted_frames = sorted(frameIds, key=lambda frameId: len(track_db.get_track_ids_for_frame(frameId)))

    # Calculate the index of the median frame
    median_index = int(len(sorted_frames) * percentage)

    # Select the keyframes from the median frame to the last frame
    key_frames = sorted_frames[median_index:]

    return key_frames


def get_bundle_windows(key_frames):
    return [(key_frames[i-1], key_frames[i]) for i in range(1, len(key_frames))]


def create_factor_graph(bundle_window_frameIds):
    """
    Creates the factor graph for the bundle window
    """
    cam_pose = None
    # Compute the first frame's extrinsic matrix that maps points from camera coordinates to world coordinates
    first_cam_pose = extinsic_to_global(track_db.get_extrinsic_matrix_by_frameId(bundle_window_frameIds[0]))
    bundle_starts_in_frame_id = bundle_window_frameIds[0]
    bundle_ends_in_frame_id = bundle_window_frameIds[-1]
    # Initialize the factor graph and values
    factor_graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    # Create factors and values for each frame
    for i, frameId in enumerate(bundle_window_frameIds):
        # Create camera symbol and update values dictionary
        cam_pose_sym = gtsam.symbol(CAMERA, frameId)
        cur_cam_pose = extinsic_to_global(compose(first_cam_pose, track_db.get_extrinsic_matrix_by_frameId(frameId)))
        cam_pose = gtsam.Pose3(cur_cam_pose)
        initial_estimates.insert(cam_pose_sym, gtsam.Pose3(cam_pose))

        if i == 0:
            # Add prior factor for the first frame
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
            factor_graph.add(gtsam.PriorFactorPose3(cam_pose_sym, cam_pose, prior_noise))


    relevant_tracks = track_db.get_tracks_in_bundle_window(bundle_starts_in_frame_id, bundle_ends_in_frame_id)

    # For each relevant track, create measurement factors
    for track_id in relevant_tracks:
        track_data = track_db.get_track_data(track_id)
        track_ends_in_frame_id = track_data[-1][-1]
        if track_ends_in_frame_id < bundle_ends_in_frame_id:
            continue

        # Create measurement factor for this track point
        last_frame_pose = gtsam.StereoCamera(cam_pose, GTSAM_K)

        # Track's locations in frames_in_window
        locations = track_db.get_track_locations_in_segment(track_id, bundle_starts_in_frame_id, bundle_ends_in_frame_id)

        # # Track's location at the Last frame for triangulations
        # last_frame_location = track_locations[-1]

        # Get the 2D point in the last frame for triangulation
        last_point2 = gtsam.StereoPoint2(locations[-1][0], locations[-1][1], locations[-1][2])
        last_point3 = last_frame_pose.backproject(last_point2)

        point_symbol = gtsam.symbol(POINT, track_id)
        initial_estimates.insert(point_symbol, last_point3)

        for i, frame_id in enumerate(bundle_window_frameIds):
            cam_symbol = gtsam.symbol(CAMERA, frame_id)
            # Get the measured 2D point in the current frame
            measured_point2 = gtsam.StereoPoint2(locations[i][0], locations[i][1], locations[i][2])

            # Create the factor between the measured and projected points
            stereomodel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([ 0.1, 0.1, 0.1]))

            factor = gtsam.GenericStereoFactor3D(measured_point2, stereomodel_noise, cam_symbol, point_symbol, GTSAM_K)

            # Add the factor to the factors list
            factor_graph.add(factor)

        # create_measurement_factor(track_id, track_point)
    return factor_graph, initial_estimates

def q2():
    # Step 1: Select Keyframes
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    key_frames = criteria(frameIds=frameIds)
    bundle_windows = get_bundle_windows(key_frames)
    # Step 2: Define Bundle Optimization
    first_window_frameIds = bundle_windows[0]

    # Step 3: Add Factors and Initial Estimates for Keyframes and Landmarks
    bundle_graph, initial_estimates = create_factor_graph(first_window_frameIds)

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
    gtsam.utils.plot.plot_trajectory(fignum=0, values=optimized_estimates, title="!")
    plt.show()


def compute_factor_error(factor, values):
    """
    Compute the factor error given a factor and a values dictionary.
    """
    error = factor.unwhitenedError(values)
    return np.linalg.norm(error)

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
    first_frame_camera_matrix = track_db.get_extrinsic_matrix_by_frameId(frameIds[0])
    last_camera = compose(first_frame_camera_matrix, last_camera_matrix)
    last_camera_global = extinsic_to_global(last_camera)

    last_cam_pose = gtsam.Pose3(last_camera_global)
    last_frame_pose = gtsam.StereoCamera(last_cam_pose, GTSAM_K)
    last_point2 = gtsam.StereoPoint2(x_l_last, x_r_last,y_last)  # Get the 2D point in the last frame for triangulation
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
        measured_point2 = gtsam.StereoPoint2(locations[i][0],locations[i][1],locations[i][2])

        # Project the 3D point to the current frame
        frame_pose = gtsam.StereoCamera(frame_pose, GTSAM_K)
        projected_point2 = frame_pose.project(last_point3)

        # Create the factor between the measured and projected points
        stereomodel_noise = stereomodel(1, 1)
        factor = gtsam.GenericStereoFactor3D(measured_point2,stereomodel_noise, cam_symbol, point_symbol, GTSAM_K)

        # Add the factor to the factors list
        factors.append(factor)

        # Compute and store the reprojection error
        measured_point2 = measured_point2.uL(), measured_point2.uR(), measured_point2.v()
        projected_point2 = projected_point2.uL(), projected_point2.uR(), projected_point2.v()
        reprojection_errors.append(np.linalg.norm(np.array(measured_point2) - np.array(projected_point2)))

    return reprojection_errors, factors, values


def get_pose_for_frameId(first_frame_camera_matrix, frame_id):
    ex_mat = extinsic_to_global(track_db.get_extrinsic_matrix_by_frameId(frame_id))
    camera_wrt_first_frame = compose(first_frame_camera_matrix, ex_mat)
    current_pose = extinsic_to_global(camera_wrt_first_frame)
    frame_pose = gtsam.Pose3(current_pose)
    return frame_pose




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

