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

    # Plot the factor errors
    factor_errors = [compute_factor_error(factor, values) for factor in factors]
    plt.plot(range(len(frames)), factor_errors)
    plt.xlabel('Frame Index')
    plt.ylabel('Factor Error')
    plt.title('Factor Error over Track')
    plt.show()

def compose_transformations(first_ex_mat, second_ex_mat):
    """
    Compute the composition of two extrinsic camera matrices.
    first_ex_mat : A -> B
    second_ex_mat : B -> C
    composed_ex_mat : A -> C
    """
    composed_ex_mat = np.dot(second_ex_mat[:, :3], first_ex_mat[:, :3])
    composed_ex_mat[:, 3] = np.dot(second_ex_mat[:, :3], first_ex_mat[:, 3]) + second_ex_mat[:, 3]
    return composed_ex_mat

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
    gtsam_calib_mat = GTSAM_K  # Set your calibration matrix here

    # Triangulate 3D point from the last frame
    last_camera_matrix = track_db.get_extrinsic_matrix_by_frameId(frameIds[-1])
    first_frame_camera_matrix = track_db.get_extrinsic_matrix_by_frameId(frameIds[0])
    first_camera = compose(first_frame_camera_matrix, last_camera_matrix)
    first_camera_pose = extinsic_to_global(first_camera)

    last_cam_pose = gtsam.Pose3(first_camera_pose)
    last_frame_pose = gtsam.StereoCamera(last_cam_pose, gtsam_calib_mat)
    last_point2 = gtsam.StereoPoint2(x_l_last, x_r_last,y_last)  # Get the 2D point in the last frame for triangulation
    last_point3 = last_frame_pose.backproject(last_point2)

    # Add the 3D point to the values dictionary
    point_symbol = gtsam.symbol(POINT, 0)
    values.insert(point_symbol, last_point3)
    current_transformations_from_first_frame = lambda mat: compose(first_frame_camera_matrix, mat)
    for i, frame_id in enumerate(frameIds):
        ex_mat = extinsic_to_global(track_db.get_extrinsic_matrix_by_frameId(frame_id))
        camera_wrt_first_frame = current_transformations_from_first_frame(ex_mat)
        current_pose = extinsic_to_global(camera_wrt_first_frame)
        frame_pose = gtsam.Pose3(current_pose)

        # Insert left pose symbol for the current frameId:
        cam_symbol = gtsam.symbol(CAMERA, frame_id)
        values.insert(cam_symbol, frame_pose)

        # Get the measured 2D point in the current frame
        measured_point2 = gtsam.StereoPoint2(locations[i][0],locations[i][1],locations[i][2])

        # Project the 3D point to the current frame
        frame_pose = gtsam.StereoCamera(frame_pose, gtsam_calib_mat)
        projected_point2 = frame_pose.project(last_point3)

        # Create the factor between the measured and projected points
        stereomodel_noise = stereomodel(1, 1)
        factor = gtsam.GenericStereoFactor3D(measured_point2,stereomodel_noise, cam_symbol, point_symbol, gtsam_calib_mat)

        # Add the factor to the factors list
        factors.append(factor)

        # Compute and store the reprojection error
        measured_point2 = measured_point2.uL(), measured_point2.uR(), measured_point2.v()
        projected_point2 = projected_point2.uL(), projected_point2.uR(), projected_point2.v()
        reprojection_errors.append(np.linalg.norm(np.array(measured_point2) - np.array(projected_point2)))

    return reprojection_errors, factors, values

if __name__ == "__main__":
    random.seed(6)
    s = gtsam.StereoCamera()
    track_db = TrackDatabase()
    deserialization_result = track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
    if deserialization_result == FAILURE:
        _, track_db = track_camera_for_many_images()
        track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)

    q1()

