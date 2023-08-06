import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

import utils.utils
from models.Constants import *
from models.TrackDatabase import TrackDatabase
from utils.plotters import plot_trajectories, plot_localization_error_over_time
from utils.utils import invert_Rt_transformation, get_gt_trajectory
import matplotlib.pyplot as plt


compose = lambda first_matrix, last_matrix: last_matrix @ np.append(first_matrix, [np.array([0, 0, 0, 1])], axis=0)
K, M1, M2 = utils.utils.read_cameras()
GTSAM_K = utils.utils.get_gtsam_calib_mat(K, M2)

def get_relative_transformation_same_source_cs(T1, T2):
    # returning T from T1 dest cs to T2 dest cs.
    T1, T2 = np.copy(T1), np.copy(T2)
    R1, t1 = T1[:, :-1], T1[:, -1]
    R2, t2 = T2[:, :-1], T2[:, -1]
    res = np.zeros_like(T1)
    res[:,:-1] = R2@R1.T
    res[:,-1] = t2 - R2@R1.T@t1
    return res


def alt_compose(T_a_c, T_a_b):
    R1, t1 = T_a_b[:, :-1], T_a_b[:, -1]
    R2, t2 = T_a_c[:, :-1], T_a_c[:, -1]
    R_c_b = R2@R1
    t_c_b = (R2@t1) + t2
    res = np.zeros_like(T_a_c)
    res[:, :-1] = R_c_b
    res[:, -1] = t_c_b
    return res


def get_bundle_windows(key_frames):
    return [(key_frames[i - 1], key_frames[i]) for i in range(1, len(key_frames))]


def get_translation_rotation_diff(pose_i, pose_j):
    translation_i = pose_i[:3, 3]  # Extract translation vector from pose_i
    translation_j = pose_j[:3, 3]  # Extract translation vector from pose_j

    distance = np.linalg.norm(translation_j - translation_i)
    rotation_i = Rotation.from_matrix(pose_i[:3, :3])  # Extract rotation matrix from pose_i and convert to a rotation object
    rotation_j = Rotation.from_matrix(pose_j[:3, :3])  # Extract rotation matrix from pose_j and convert to a rotation object

    rotation_diff = rotation_i.inv() * rotation_j  # Compute the relative rotation difference between pose_i and pose_j
    axis_angle = rotation_diff.as_rotvec()
    # sign = 1 if np.sum(np.sign(axis_angle)) == (3 or -3) else -1 # Sign of the angle
    return distance, rotation_diff.magnitude()


def select_keyframes_by_track_max_distance(frames, track_db):
    frameIds = [0]
    frameId = 1
    rotation_accumulator = 0
    while frameId < len(frames):
        cur_pose = track_db.get_extrinsic_matrix_by_frameId(frameId)
        prev_pose = track_db.get_extrinsic_matrix_by_frameId(frameIds[-1])
        distance, angle = get_translation_rotation_diff(prev_pose, cur_pose)
        window_size = frameId-frameIds[-1]
        rotation_accumulator += angle

        if window_size == 10 or abs(rotation_accumulator) > .05 or abs(angle)>0.03:
            frameIds.append(frameId)
            rotation_accumulator = 0

        frameId += 1

    return frameIds


def create_factor_graph(track_db, bundle_starts_in_frame_id, bundle_ends_in_frame_id):
    """
    Creates the factor graph for the bundle window
    """

    landmarks = set()
    # Compute the first frame's extrinsic matrix that maps points from camera coordinates to world coordinates
    first_cam_pose = track_db.get_extrinsic_matrix_by_frameId(bundle_starts_in_frame_id)
    #relevant_tracks = get_only_relevant_tracks(track_db, bundle_starts_in_frame_id)
    relevant_tracks = get_only_relevant_tracks_all(track_db, bundle_starts_in_frame_id, bundle_ends_in_frame_id)
    frameId_to_cam_pose, factor_graph, initial_estimates = init_factor_graph_variables(track_db, bundle_ends_in_frame_id,
                                                                    bundle_starts_in_frame_id,
                                                                            first_cam_pose)
    # Get all the tracks that related to this specific bundle window

    # For each relevant track, create measurement factors
    for track_data, trackId in relevant_tracks:
        # Check whether the track is too short
        track_ends_in_frame_id = track_data[LAST_ITEM][FRAME_ID]
        track_starts_in_frame_id = track_data[0][FRAME_ID]
        # if track_ends_in_frame_id < bundle_ends_in_frame_id or bundle_starts_in_frame_id < track_starts_in_frame_id:
        track_length_in_bundle = min(track_ends_in_frame_id, bundle_ends_in_frame_id) - max(bundle_starts_in_frame_id, track_starts_in_frame_id) + 1
        if track_length_in_bundle < 5:
            continue
        # if track_ends_in_frame_id < bundle_ends_in_frame_id:
        #     continue

        # TODO original code uses a 2d point coordinates from last frame of track, but uses the cam_pose of the last frame in the bundle... wrong
        # Create measurement factor for this track point
        offset = track_data[0][FRAME_ID]
        frameId_of_last_frame_of_track_in_bundle = min(bundle_ends_in_frame_id, track_ends_in_frame_id)
        frameId_of_first_frame_of_track_in_bundle = max(bundle_starts_in_frame_id, track_starts_in_frame_id)
        cam_pose_of_last_frame = frameId_to_cam_pose[frameId_of_last_frame_of_track_in_bundle]
        last_frame_pose = gtsam.StereoCamera(cam_pose_of_last_frame, GTSAM_K)
        index_of_last_relevant_track_point = frameId_of_last_frame_of_track_in_bundle - offset
        last_loc = track_data[index_of_last_relevant_track_point][LOCATIONS_IDX]
        # Get the 2D point in the last frame for triangulation
        last_point2 = gtsam.StereoPoint2(last_loc[0], last_loc[1], last_loc[2])

        last_point3 = last_frame_pose.backproject(last_point2)

        # if last_point3[2] <= 0 or last_point3[2] >= 150:
        #     continue

        # find the z value from the first frame and filter by it ...
        cam_pose_of_first_frame = frameId_to_cam_pose[frameId_of_first_frame_of_track_in_bundle]
        first_frame_pose = gtsam.StereoCamera(cam_pose_of_first_frame, GTSAM_K)
        index_of_first_relevant_track_point = frameId_of_first_frame_of_track_in_bundle - offset
        first_loc = track_data[index_of_first_relevant_track_point][LOCATIONS_IDX]
        first_point2 = gtsam.StereoPoint2(first_loc[0], first_loc[1], first_loc[2])
        first_point3 = first_frame_pose.backproject(first_point2)

        if first_point3[2] < 0 or first_point3[2] >= 100:
            continue

        point_symbol = gtsam.symbol(POINT, trackId)
        initial_estimates.insert(point_symbol, last_point3)
        landmarks.add(point_symbol)
        for i, frame_id in enumerate(range(frameId_of_first_frame_of_track_in_bundle, frameId_of_last_frame_of_track_in_bundle+1)):
            cam_symbol = gtsam.symbol(CAMERA, frame_id)
            index_of_relevant_track_point = frame_id - offset
            location = track_data[index_of_relevant_track_point][LOCATIONS_IDX]
            # Get the measured 2D point in the current frame
            measured_point2 = gtsam.StereoPoint2(location[0], location[1], location[2])

            # Create the factor between the measured and projected points
            stereomodel_noise = gtsam.noiseModel.Isotropic.Sigma(3,0.1)
            factor = gtsam.GenericStereoFactor3D(measured_point2, stereomodel_noise, cam_symbol, point_symbol, GTSAM_K)

            # Add the factor to the factors list
            factor_graph.add(factor)

        # create_measurement_factor(track_id, track_point)
    return factor_graph, initial_estimates, landmarks


def init_factor_graph_variables(track_db, bundle_ends_in_frame_id, bundle_starts_in_frame_id, first_cam_pose):
    # Initialize the factor graph and values
    factor_graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    frameId_to_cam_pose = dict()
    # Create factors and values for each frame
    for i, frameId in enumerate(range(bundle_starts_in_frame_id, bundle_ends_in_frame_id + 1)):
        # Create camera symbol and update values dictionary
        # print(f"{i}'th iter where frameId: {frameId} in window:{bundle_starts_in_frame_id},{bundle_ends_in_frame_id}")
        cam_pose_sym = gtsam.symbol(CAMERA, frameId)
        # TODO I think that the use of Compose was wrong here...
        #cur_cam_pose = invert_Rt_transformation(compose(first_cam_pose, track_db.get_extrinsic_matrix_by_frameId(frameId)))
        cur_cam_pose = get_relative_transformation_same_source_cs(track_db.get_extrinsic_matrix_by_frameId(frameId), first_cam_pose)
        cam_pose = gtsam.Pose3(cur_cam_pose)
        initial_estimates.insert(cam_pose_sym, cam_pose)
        frameId_to_cam_pose[frameId] = cam_pose
        if i == 0:
            # Add prior factor for the first frame
            s= 0.1*np.array([(3 * np.pi / 180),(3 * np.pi / 180),(3 * np.pi / 180)] + [.1, 0.01, 1.0])
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(s)
            factor_graph.add(gtsam.PriorFactorPose3(cam_pose_sym, cam_pose, prior_noise))
    return frameId_to_cam_pose, factor_graph, initial_estimates


def key_frames_by_percentile(frames, percentage, track_db):
    """
    Choose keyframes by the median track len's from the last frame
    """
    key_frames = [0]
    n = len(frames)
    while key_frames[-1] < len(frames) - 1:
        last_key_frame = key_frames[-1]
        frame = frames[last_key_frame]
        tracks = track_db.get_track_ids_for_frame(frame)

        tracks_by_lengths = sorted([track_db.get_track_data(trackId)[-1][FRAME_ID] for trackId in tracks if len(track_db.get_track_data(trackId))>2])
        new_key_frame = tracks_by_lengths[int(len(tracks_by_lengths) * percentage)]
        distance, rotation = 0, 0
        for fid in range(frame, new_key_frame):
            cur_pose = track_db.get_extrinsic_matrix_by_frameId(frame)
            prev_pose = track_db.get_extrinsic_matrix_by_frameId(frame - 1)
            cur_step, angle = get_translation_rotation_diff(prev_pose, cur_pose)
            distance += cur_step
            rotation += angle
            if (abs(rotation) > 0.5 or abs(distance) > 100) and 4 < (fid - frame) < 20:
                new_key_frame = fid
                break

        key_frames.append(min(new_key_frame, n - 1))

    return key_frames

def key_frames_by_elapsed_time(track_db, step):
    frameIds = list(track_db.get_frameIds())
    key_frames = [frameIds[i] for i in range(0, len(frameIds), step)]
    if 15 > (frameIds[-1] - key_frames[-1]) > 5:
        key_frames.append(frameIds[-1])
    return key_frames


def get_only_relevant_tracks_all(track_db, starting_frame_id, ending_frame_id):
    tracksIds = set()
    for frameId in range(starting_frame_id, ending_frame_id+1):
        tracksIds.update(set(track_db.get_track_ids_for_frame(frameId)))
    relevant_tracks = [(track_db.get_track_data(trackId), trackId) for trackId in tracksIds]

    return relevant_tracks

def get_only_relevant_tracks(track_db, frame_id):
    tracksIds = set()
    tracksIds.update(set(track_db.get_track_ids_for_frame(frame_id)))
    relevant_tracks = [(track_db.get_track_data(trackId), trackId) for trackId in tracksIds]

    return relevant_tracks


def triangulate_and_project(track,track_db):
    """
    Triangulate a 3D point from the last frame of the track
    and project it to all frames in the track.
    Return the reprojection errors and factors.
    """

    frameIds = [track_feature[FRAME_ID] for track_feature in track]
    points = [track_feature[LOCATIONS_IDX] for track_feature in track]
    x_l_last, x_r_last, y_last = points[-1]
    values = gtsam.Values()

    # Create a StereoCamera for each frame in the track
    stereo_cameras = get_stereoCam_per_frame(frameIds,track_db)

    # Triangulate 3D point in global coordinates from the last frame
    image_point = gtsam.StereoPoint2(x_l_last, x_r_last, y_last)  # (x_left, x_right, y) Image point in the last frame
    stereo_camera_last = stereo_cameras[-1]
    triangulated_point = stereo_camera_last.backproject(image_point)

    # Project the 3D point to all frames in the track
    reprojected_points = reproject_points_to_last(frameIds, stereo_cameras, triangulated_point)

    p_sym = gtsam.symbol(POINT, 0)
    values.insert(p_sym, triangulated_point)
    # Calculate and plot the re-projection error size over the track's images
    factors, reprojection_errors, values = compute_reprojection_errors(frameIds, points, reprojected_points,
                                                                       stereo_cameras, values, p_sym)

    return reprojection_errors, factors, values


def compute_reprojection_errors(frameIds, points, reprojected_points, stereo_cameras, values, point_sym):
    reprojection_errors = []
    factors = []
    for i, frameId in enumerate(frameIds):
        cam_symbol = gtsam.symbol(CAMERA, frameId)
        image_point = points[i]
        reprojected_point = reprojected_points[i]
        error = np.linalg.norm(
            np.array(image_point) - np.array([reprojected_point.uL(), reprojected_point.uR(), reprojected_point.v()]))
        reprojection_errors.append(error)
        values.insert(cam_symbol, stereo_cameras[i].pose())

        measured_point = gtsam.StereoPoint2(points[i][0], points[i][1], points[i][2])
        stereo_model_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1)
        factor = gtsam.GenericStereoFactor3D(measured_point, stereo_model_noise,
                                             cam_symbol, point_sym, GTSAM_K)
        factors.append(factor)
    return factors, reprojection_errors, values


def calculate_reprojection_errors(frameIds, points, reprojected_points):
    reprojection_errors = []
    for i, frame in enumerate(frameIds):
        image_point = points[i]
        reprojected_point = reprojected_points[i]
        error = np.linalg.norm(
            np.array(image_point) - np.array([reprojected_point.uL(), reprojected_point.uR(), reprojected_point.v()]))
        reprojection_errors.append(error)
    return reprojection_errors


def reproject_points_to_last(frameIds, stereo_cameras, triangulated_point):
    reprojected_points = []
    for i, frameId in enumerate(frameIds):

        stereo_camera = stereo_cameras[i]
        reprojected_point = stereo_camera.project(triangulated_point)
        reprojected_points.append(reprojected_point)

    return reprojected_points


def get_stereoCam_per_frame(frameIds,track_db):
    stereo_cameras = []
    for frame in frameIds:
        # Use the global camera matrices calculated in exercise 3
        extrinsic_matrix = track_db.get_extrinsic_matrix_by_frameId(frame)
        # Create StereoCamera using global camera matrices
        stereo_camera = gtsam.StereoCamera(gtsam.Pose3(invert_Rt_transformation(extrinsic_matrix)), GTSAM_K)
        stereo_cameras.append(stereo_camera)
    return stereo_cameras


def get_pose_for_frameId(first_frame_camera_matrix, frame_id, track_db):
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
    ex_mat = invert_Rt_transformation(track_db.get_extrinsic_matrix_by_frameId(frame_id))

    # Compute the camera pose relative to the first frame
    camera_wrt_first_frame = compose(first_frame_camera_matrix, ex_mat)
    current_pose = invert_Rt_transformation(camera_wrt_first_frame)

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
    return factor.error(values)


def solve_one_bundle(track_db, bundle_window, debug=True):
    # Add Factors and Initial Estimates for Keyframes and Landmarks
    bundle_starts_in_frame_id, bundle_ends_in_frame_id = bundle_window
    bundle_graph, initial_estimates, landmarks = create_factor_graph(track_db, bundle_starts_in_frame_id,
                                                                     bundle_ends_in_frame_id)

    # Perform Bundle Adjustment Optimization
    optimizer = gtsam.LevenbergMarquardtOptimizer(bundle_graph, initial_estimates)
    optimized_estimates = optimizer.optimize()
    try:
        bundle_covariance = gtsam.Marginals(bundle_graph, optimized_estimates)
    except RuntimeError as e:
        print('\033[91m' + f"Caught Exception When Calculating Marginals: {e}" + '\033[0m')
        bundle_starts_in_frame_id, bundle_ends_in_frame_id = bundle_window
        bundle_graph, initial_estimates, landmarks = create_factor_graph(track_db, bundle_starts_in_frame_id,
                                                                         bundle_ends_in_frame_id)
        raise e
    # Print Total Factor Graph Error
    if debug:
        initial_error = bundle_graph.error(initial_estimates)
        optimized_error = bundle_graph.error(optimized_estimates)
        print("Initial Total Factor Graph Error:", initial_error)
        print("Optimized Total Factor Graph Error:", optimized_error)

    return bundle_graph, initial_estimates, landmarks, optimized_estimates, bundle_covariance



############################### Code Used for exercise 6 ###############################

def load_bundle_results(path=PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS, force_recompute=False, debug=True, track_db_path=PATH_TO_SAVE_TRACKER_FILE):
    import pickle
    if force_recompute or path is None:
        print("Recomputing Bundle Adjustment ...")
        return bundle_adjustment(path_to_serialize=path, debug=debug, plot_results=True, track_db_path=track_db_path)
    try:
        with open(path, 'rb') as f:
            bundle_adjustment_results = pickle.load(f)
            bundle_results = bundle_adjustment_results["bundle_results"]
            optimized_relative_keyframes_poses = bundle_adjustment_results["optimized_relative_keyframes_poses"]
            optimized_global_keyframes_poses = bundle_adjustment_results["optimized_global_keyframes_poses"]
            bundle_windows = bundle_adjustment_results["bundle_windows"]
            cond_matrices = bundle_adjustment_results["cond_matrices"]
            print(f"Found File At {path}, loaded bundle adjustment results...")
    except FileNotFoundError:
        print(f"No File Exists In {path}")
        print("Recomputing Bundle Adjustment ...")
        return bundle_adjustment(path_to_serialize=path, debug=debug, plot_results=True, track_db_path=track_db_path)
    return bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, cond_matrices


def get_conditional_covariance_matrix_between_keyframes(window, marginals):
    # Define the window of interest.
    ci, ck = window

    # Initialize an empty KeyVector object to store keys associated with cameras ci and ck.
    key_vectors = gtsam.KeyVector()

    # Append keys associated with the cameras to the key vector.
    key_vectors.append(gtsam.symbol(CAMERA, ci))
    key_vectors.append(gtsam.symbol(CAMERA, ck))

    # calculate the marginal information matrix of the two cameras
    information_mat_cick = marginals.jointMarginalInformation(key_vectors).fullMatrix()

    # deduce the conditional covariance of p(ck|c0) from it
    conditional_covariance = np.linalg.inv(information_mat_cick[6:, 6:])

    return conditional_covariance


def get_relative_pose_between_frames(frame_a, frame_b, estimates):
    # returns the transformation from frame_b world into frame_a world
    # Retrieve optimized poses for cameras ci and ck.
    camera_a = estimates.atPose3(gtsam.symbol(CAMERA, frame_a))
    camera_b = estimates.atPose3(gtsam.symbol(CAMERA, frame_b))

    # Calculate the relative pose from camera ci to camera ck.
    relative_pose = camera_a.between(camera_b)
    return relative_pose


def create_pose_graph(bundle_results, rel_poses_lst, optimized_global_keyframes_poses, rel_cov_mat_lst):
    pose_graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    landmarks = set()
    # Initialize first camera location and prior factor
    _, bundle_window, bundle_graph, _, _, optimized_estimates = bundle_results[0]
    first_camera_pose = optimized_global_keyframes_poses[0]
    c0 = gtsam.symbol(CAMERA, bundle_window[0])
    landmarks.add(c0)
    s = np.array([1/18,1/18,.1/18] + [.1, 0.01, 1.0]) * 0.1
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(s)
    pose_graph.add(gtsam.PriorFactorPose3(c0, first_camera_pose, prior_noise))
    initial_estimates.insert(c0, first_camera_pose)

    # Initialize the rest of variables and factors
    for bundle_res in bundle_results:
        i, bundle_window, bundle_graph, _, landmarks, optimized_estimates = bundle_res
        c0 = gtsam.symbol(CAMERA, bundle_window[0])
        c1 = gtsam.symbol(CAMERA, bundle_window[1])
        landmarks.add(c1)
        c0_global_pose = optimized_global_keyframes_poses[i]
        c1_global_pose = optimized_global_keyframes_poses[i + 1]
        initial_estimates.insert(c1, c1_global_pose)

        #relative_pose = get_relative_pose_between_frames(bundle_window[0], bundle_window[1], optimized_estimates)
        relative_pose = rel_poses_lst[i]
        noise_cov = gtsam.noiseModel.Gaussian.Covariance(rel_cov_mat_lst[i] *.1)
        pose_factor = gtsam.BetweenFactorPose3(c0, c1, relative_pose, noise_cov)
        pose_graph.add(pose_factor)
    return pose_graph, initial_estimates, landmarks


def bundle_adjustment(path_to_serialize=None, debug=False, plot_results=None, track_db=None, track_db_path=PATH_TO_SAVE_TRACKER_FILE):
    """
    This function solves bundle adjustment for many bundles independently, and calculates the resulting trajectory.
    :param path_to_serialize: path in which the bundle results will be saved at.
    :param debug: if True execution will include printouts.
    :param plot_results: if True a plot of the trajectory in comparison to the Ground Truth will be presented.
    :param track_db: The tracking data.
    :return: Results of bundle adjustment
    """
    PATH_TO_SAVE_COMPARISON_TO_GT = "plots/compare_to_ground_truth"
    PATH_TO_SAVE_LOCALIZATION_ERROR = "plots/localization_error"
    PATH_TO_SAVE_2D_TRAJECTORY = "plots/2d_view_of_the_entire_scene"

    # Step 1: Initialize TrackDatabase and Select meaningful Keyframes
    if track_db is None:
        track_db = TrackDatabase()

    deserialization_result = track_db.deserialize(track_db_path)
    if deserialization_result == FAILURE:
        _, track_db = utils.utils.track_camera_for_many_images()
        track_db.serialize(track_db_path)
        print("serialization is done!")

    frameIds = track_db.get_frameIds()
    #key_frames = key_frames_by_percentile(list(frameIds.keys()), .89, track_db)
    key_frames = key_frames_by_elapsed_time(track_db, step=20)
    bundle_windows = get_bundle_windows(key_frames)
    prev = key_frames[0]
    for frame_id in key_frames[1:]:
        if frame_id - prev < 5:
            print(frame_id - prev)
    # Step 2: Solve Every Bundle Window
    num_factor_in_bundles = []
    bundle_results = []
    cond_matrices = []
    for i, bundle_window in enumerate(bundle_windows):

        bundle_graph, initial_estimates, landmarks, optimized_estimates, marginals = solve_one_bundle(track_db, bundle_window, debug=False)
        cond_mat = get_conditional_covariance_matrix_between_keyframes(bundle_window, marginals)
        cond_matrices.append(cond_mat)
        bundle_results.append((i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates))
        num_factor_in_bundles.append(bundle_graph.size())
        if debug:
            initial_error = bundle_graph.error(initial_estimates)
            optimized_error = bundle_graph.error(optimized_estimates)
            print(f"----------------Solved bundle # {i} from frame {bundle_window[0]} to frame {bundle_window[1]}------------------")
            print(f"number of factors in graph is {num_factor_in_bundles[-1]}")
            print(f"initial graph error was {initial_error}, and after optimization the error is {optimized_error}")
    if debug:
        print(f"FINISHED SOLVING BUNDLES!")
        print(f"Mean number of factors per graph is: {np.mean(num_factor_in_bundles)})")
        print(f"Min number of factors over graphs: {np.min(num_factor_in_bundles)})")
        print(f"Max number of factors over graphs: {np.max(num_factor_in_bundles)})")

    # Step 3: Extracting Relative Poses Between Consecutive KeyFrames, and Also their Global Pose (relative to frame 0)
    optimized_relative_keyframes_poses = []
    optimized_global_keyframes_poses = []
    _, bundle_window, _, _, _, optimized_estimates = bundle_results[0]
    estimated_camera_position = optimized_estimates.atPose3(gtsam.symbol(CAMERA, bundle_window[0]))
    optimized_global_keyframes_poses.append(estimated_camera_position)
    global_3d_points = []
    for bundle_res in bundle_results:
        i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates = bundle_res
        estimated_camera_position = optimized_estimates.atPose3(gtsam.symbol(CAMERA, bundle_window[1]))  # transforms from end of bundle to its beginning
        optimized_relative_keyframes_poses.append(estimated_camera_position)
        previous_global_pose = optimized_global_keyframes_poses[-1]  # transforms from beginning of bundle to global world
        current_global_pose = previous_global_pose * estimated_camera_position # transforms from end of bundle to global world
        bundle_3d_points = gtsam.utilities.extractPoint3(optimized_estimates)
        for point in bundle_3d_points:
            global_point = previous_global_pose.transformFrom(gtsam.Point3(point))
            global_3d_points.append(global_point)
        optimized_global_keyframes_poses.append(current_global_pose)

    if plot_results:
        # Compare Our Results With The Ground Truth Trajectory
        global_3d_points_numpy = np.array(global_3d_points)
        global_Rt_poses_in_numpy = np.array([pose.translation() for pose in optimized_global_keyframes_poses])
        gt_camera_positions = get_gt_trajectory()[np.array(key_frames)]
        plot_trajectories(camera_positions=global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions,
                          points_3d=global_3d_points_numpy, path=PATH_TO_SAVE_COMPARISON_TO_GT)

        # Step 6: Presenting KeyFrame Localization Over Time
        plot_localization_error_over_time(key_frames, camera_positions=global_Rt_poses_in_numpy,
                                          gt_camera_positions=gt_camera_positions, path=PATH_TO_SAVE_LOCALIZATION_ERROR)
        plt.savefig(PATH_TO_SAVE_LOCALIZATION_ERROR)

        # Step 7: Presenting a View From Above (in 2d) of the Scene, With Keyframes and 3d Points
        plot_trajectories(camera_positions=global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions,
                          path=PATH_TO_SAVE_COMPARISON_TO_GT)
        plt.savefig(PATH_TO_SAVE_2D_TRAJECTORY)

    if path_to_serialize:
        import pickle
        bundle_adjustment_results_dict = {"bundle_results": bundle_results,
                                          "optimized_relative_keyframes_poses": optimized_relative_keyframes_poses,
                                          "optimized_global_keyframes_poses": optimized_global_keyframes_poses,
                                          "bundle_windows": bundle_windows,
                                          "cond_matrices": cond_matrices}
        with open(path_to_serialize, 'wb') as f:
            pickle.dump(bundle_adjustment_results_dict, f)
    return bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, cond_matrices
