import random
import cv2
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from VAN_ex.code.exercise_2 import least_squares
from VAN_ex.code.exercise_3 import get_gt_trajectory
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from utils import utils
from utils.plotters import draw_3d_points, draw_inlier_and_outlier_matches, draw_matches, plot_four_cameras, \
    draw_supporting_matches, plot_trajectories, plot_regions_around_matching_pixels, plot_dict, plot_connectivity_graph, \
    gen_hist, plot_reprojection_errors, plot_localization_error_over_time
from utils.utils import *
from matplotlib import pyplot as plt
from models.Constants import *
import gtsam
from gtsam.utils import plot

PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker"
K, M1, M2 = utils.read_cameras()
GTSAM_K = utils.get_gtsam_calib_mat(K, M2)

compose = lambda first_matrix, last_matrix: last_matrix @ np.append(first_matrix, [np.array([0, 0, 0, 1])], axis=0)

def get_relative_transformation_same_source_cs(T1, T2):
    # returning T from T1 dest cs to T2 dest cs.
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


def choose_key_frames_by_elapsed_time(frameIds, num_frames_per_bundle=25):
    """
    choosing proper keyframes using time elapsed criterion
    """
    min_frameId, max_frameId = min(frameIds.keys()), max(frameIds.keys())
    key_frames = [n for n in range(min_frameId, max_frameId + 1, num_frames_per_bundle-1)]
    if 10 > (max_frameId - key_frames[-1]) > 0:
        key_frames[-1] = max_frameId
    elif (max_frameId - key_frames[-1]) >= 10:
        key_frames.append(max_frameId)
    return key_frames


def choose_key_frames_by_tracks_count_median(frameIds, percentage=.82):
    """
    choosing proper keyframes using a median criterion
    """
    # Sort frames based on track length (ascending order)
    sorted_frames = sorted(frameIds, key=lambda frameId: len(track_db.get_track_ids_for_frame(frameId)))
    for fid in sorted_frames:
        l = len(track_db.get_track_ids_for_frame(fid))
        #print(f"frame id {fid} has {l} tracks")
    # Calculate the index of the median frame
    median_index = int(len(sorted_frames) * percentage)

    # Select the keyframes from the median frame to the last frame
    key_frames = sorted_frames[median_index:]
    # return [frameId for frameId in range(1, len(frameIds),5)]
    return key_frames

def get_bundle_windows(key_frames):
    return [(key_frames[i - 1], key_frames[i]) for i in range(1, len(key_frames))]


def _compute_distance(pose_i, pose_j):
    translation_i = pose_i[:3, 3]  # Extract translation vector from pose_i
    translation_j = pose_j[:3, 3]  # Extract translation vector from pose_j

    distance = np.linalg.norm(translation_j - translation_i)
    return distance


def select_keyframes_by_track_max_distance(frames, min_distance_threshold=1, max_keyframes=20):
    distances = []
    for i in range(len(frames) - 1):
        frameId = frames[i]
        next_frameId = frames[i+1]
        pose_i = track_db.get_extrinsic_matrix_by_frameId(frameId)
        pose_j = track_db.get_extrinsic_matrix_by_frameId(next_frameId)
        distance = _compute_distance(pose_i, pose_j)
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


def create_factor_graph(track_db, bundle_starts_in_frame_id, bundle_ends_in_frame_id):
    """
    Creates the factor graph for the bundle window
    """
    cam_pose = None
    landmarks = set()
    # Compute the first frame's extrinsic matrix that maps points from camera coordinates to world coordinates
    first_cam_pose = track_db.get_extrinsic_matrix_by_frameId(bundle_starts_in_frame_id)

    frameId_to_cam_pose, factor_graph, initial_estimates = init_factor_graph_variables(bundle_ends_in_frame_id,
                                                                    bundle_starts_in_frame_id,
                                                                            first_cam_pose)

    # Get all the tracks that related to this specific bundle window
    relevant_tracks = get_only_relevant_tracks(bundle_ends_in_frame_id, bundle_starts_in_frame_id)

    # For each relevant track, create measurement factors
    for track_data, trackId in relevant_tracks:
        # Check whether the track is too short
        track_ends_in_frame_id = track_data[LAST_ITEM][FRAME_ID]
        # if track_ends_in_frame_id < bundle_ends_in_frame_id:
        #     continue
        # TODO original code uses a 2d point coordinates from last frame of track, but uses the cam_pose of the last frame in the bundle... wrong
        # Create measurement factor for this track point
        offset = track_data[0][FRAME_ID]
        frameId_of_last_frame_of_track_in_bundle = min(bundle_ends_in_frame_id, track_ends_in_frame_id)
        frameId_of_first_frame_of_track_in_bundle = max(bundle_starts_in_frame_id, track_data[0][FRAME_ID])
        cam_pose_of_last_frame = frameId_to_cam_pose[frameId_of_last_frame_of_track_in_bundle]
        last_frame_pose = gtsam.StereoCamera(cam_pose_of_last_frame, GTSAM_K)
        index_of_last_relevant_track_point = frameId_of_last_frame_of_track_in_bundle - offset
        last_loc = track_data[index_of_last_relevant_track_point][LOCATIONS_IDX]
        # Get the 2D point in the last frame for triangulation
        last_point2 = gtsam.StereoPoint2(last_loc[0], last_loc[1], last_loc[2])
        last_point3 = last_frame_pose.backproject(last_point2)

        point_symbol = gtsam.symbol(POINT, trackId)
        initial_estimates.insert(point_symbol, last_point3)
        landmarks.add(point_symbol)
        #locations = np.array(track_data, dtype=object)[:,1,...]
        for i, frame_id in enumerate(range(frameId_of_first_frame_of_track_in_bundle, frameId_of_last_frame_of_track_in_bundle+1)):
            cam_symbol = gtsam.symbol(CAMERA, frame_id)
            index_of_relevant_track_point = frame_id - offset
            location = track_data[index_of_relevant_track_point][LOCATIONS_IDX]
            # Get the measured 2D point in the current frame
            measured_point2 = gtsam.StereoPoint2(location[0], location[1], location[2])

            # Create the factor between the measured and projected points
            stereomodel_noise = gtsam.noiseModel.Diagonal.Sigmas(0.1*np.array([2, 2, 2]))
            factor = gtsam.GenericStereoFactor3D(measured_point2, stereomodel_noise, cam_symbol, point_symbol, GTSAM_K)

            # Add the factor to the factors list
            factor_graph.add(factor)

        # create_measurement_factor(track_id, track_point)
    return factor_graph, initial_estimates, landmarks


def init_factor_graph_variables(bundle_ends_in_frame_id, bundle_starts_in_frame_id, first_cam_pose):
    # Initialize the factor graph and values
    factor_graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    frameId_to_cam_pose = dict()
    # Create factors and values for each frame
    for i, frameId in enumerate(range(bundle_starts_in_frame_id, bundle_ends_in_frame_id + 1)):
        # Create camera symbol and update values dictionary
        cam_pose_sym = gtsam.symbol(CAMERA, frameId)
        # TODO I think that the use of Compose was wrong here...
        #cur_cam_pose = invert_Rt_transformation(compose(first_cam_pose, track_db.get_extrinsic_matrix_by_frameId(frameId)))
        cur_cam_pose = get_relative_transformation_same_source_cs(track_db.get_extrinsic_matrix_by_frameId(frameId),first_cam_pose)
        cam_pose = gtsam.Pose3(cur_cam_pose)
        initial_estimates.insert(cam_pose_sym, cam_pose)
        frameId_to_cam_pose[frameId] = cam_pose
        if i == 0:
            # Add prior factor for the first frame
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas([1, 1, 1, 2, 2, 2])
            factor_graph.add(gtsam.PriorFactorPose3(cam_pose_sym, cam_pose, prior_noise))
    return frameId_to_cam_pose, factor_graph, initial_estimates


# def get_only_relevant_tracks(bundle_ends_in_frame_id, bundle_starts_in_frame_id):
#     tracksIds = track_db.get_track_ids_for_frame(bundle_starts_in_frame_id)
#     tracks = [(track_db.get_track_data(trackId), trackId) for trackId in tracksIds]
#     # TODO don't we filter out too many tracks? what about tracks that were seen only on some of the bundle frames? and also need to make sure that we look only at relevant parts of selected tracks
#     relevant_tracks = [(track, trackId) for (track, trackId) in tracks if track[-1][-1] >= bundle_ends_in_frame_id]
#     return relevant_tracks

def get_only_relevant_tracks(bundle_ends_in_frame_id, bundle_starts_in_frame_id):
    bundle_length = 1 + bundle_ends_in_frame_id - bundle_starts_in_frame_id
    tracksIds = set()
    frameIds_in_bundle = set(range(bundle_starts_in_frame_id, bundle_ends_in_frame_id+1))
    for frame_id in range(bundle_starts_in_frame_id, bundle_ends_in_frame_id):
        tracksIds.update(set(track_db.get_track_ids_for_frame(frame_id)))
    tracks = [(track_db.get_track_data(trackId), trackId) for trackId in tracksIds]
    relevant_tracks = [(track, trackId) for (track, trackId) in tracks if len(frameIds_in_bundle.intersection(set(track_db.get_frame_ids_for_track(trackId)))) >= 7]
    return relevant_tracks

def triangulate_and_project(track):
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
    stereo_cameras = get_stereoCam_per_frame(frameIds)

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
                                                                       stereo_cameras, values)

    return reprojection_errors, factors, values


def compute_reprojection_errors(frameIds, points, reprojected_points, stereo_cameras, values):
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
                                             cam_symbol, gtsam.symbol(POINT,0), GTSAM_K)
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


def get_stereoCam_per_frame(frameIds):
    stereo_cameras = []
    for frame in frameIds:
        # Use the global camera matrices calculated in exercise 3
        extrinsic_matrix = track_db.get_extrinsic_matrix_by_frameId(frame)
        # Create StereoCamera using global camera matrices
        stereo_camera = gtsam.StereoCamera(gtsam.Pose3(invert_Rt_transformation(extrinsic_matrix)), GTSAM_K)
        stereo_cameras.append(stereo_camera)
    return stereo_cameras


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
    error = factor.unwhitenedError(values) # TODO is this the correct error function? what about .error? I would think that no need for another norm
    return np.linalg.norm(error)


def q1():
    PATH_TO_SAVE_R_ERROR_OVER_TRACK_FRAMES = "q1_r_error_per_frame"
    PATH_TO_SAVE_FACTOR_ERROR_OVER_TRACK_FRAMES = "q1_factor_error_per_frame"
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    trackId, track = track_db.get_random_track_of_length(10)
    reprojection_errors, factors, values = triangulate_and_project(track)
    frameIds = [track_feature[FRAME_ID] for track_feature in track]

    # Plot the re-projection errors
    plt.figure()
    plt.plot(frameIds, reprojection_errors)
    plt.xlabel('Frame Index')
    plt.ylabel('Re-projection Error')
    plt.title('Re-projection Error over Track')
    #plt.show()
    plt.savefig(PATH_TO_SAVE_R_ERROR_OVER_TRACK_FRAMES)

    # Plot the factor errors
    factor_errors = [compute_factor_error(factor, values) for factor in factors]
    plt.figure()
    plt.plot(frameIds, factor_errors) # TODO we were asked to plot reprojection error as function of factor error (or vice versa...)
    plt.xlabel('Frame Index')
    plt.ylabel('Factor Error')
    plt.title('Factor Error over Track')
    #plt.show()
    plt.savefig(PATH_TO_SAVE_FACTOR_ERROR_OVER_TRACK_FRAMES)


def solve_one_bundle(track_db, bundle_window, debug=True):
    # Add Factors and Initial Estimates for Keyframes and Landmarks
    bundle_starts_in_frame_id, bundle_ends_in_frame_id = bundle_window
    bundle_graph, initial_estimates, landmarks = create_factor_graph(track_db, bundle_starts_in_frame_id,
                                                                     bundle_ends_in_frame_id)

    # Perform Bundle Adjustment Optimization
    optimizer = gtsam.LevenbergMarquardtOptimizer(bundle_graph, initial_estimates)
    optimized_estimates = optimizer.optimize()

    # Print Total Factor Graph Error
    if debug:
        initial_error = bundle_graph.error(initial_estimates)
        optimized_error = bundle_graph.error(optimized_estimates)
        print("Initial Total Factor Graph Error:", initial_error)
        print("Optimized Total Factor Graph Error:", optimized_error)

    return bundle_graph, initial_estimates, landmarks, optimized_estimates


def q2():
    PATH_TO_SAVE_3D_TRAJECTORY = "q2_3d_trajectory_after_optimization"
    PATH_TO_SAVE_2D_TRAJECTORY = "q2_2d_trajectory_after_optimization" # TODO ADD THIS REQUIRED VISUALIZATION
    # Step 1: Select Keyframes
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    key_frames = choose_key_frames_by_elapsed_time(frameIds, 10)
    # select_keyframes_distance(frameIds.keys())
    bundle_windows = get_bundle_windows(key_frames)

    # Step 2: Define Bundle Optimization
    first_bundle = bundle_windows[0]
    bundle_graph, initial_estimates, landmarks, optimized_estimates = solve_one_bundle(track_db,first_bundle,debug=True)

    # Plot the Resulting Positions
    # Plotting the trajectory as a 3D graph
    bundle_starts_in_frame_id, bundle_ends_in_frame_id =  first_bundle
    cameras = np.array([initial_estimates.atPose3(gtsam.symbol(CAMERA,frameId)) for frameId in range(bundle_starts_in_frame_id, bundle_ends_in_frame_id+1)])  # List of gtsam.Pose3 objects representing poses

    # Extract X and Y coordinates from the poses
    x_coordinates = [pose.x() for pose in cameras]
    y_coordinates = [pose.y() for pose in cameras]
    z_coordinates = [pose.z() for pose in cameras]
    colormap = plt.cm.get_cmap('rainbow')


    fig = plt.figure(num=0)
    ax=fig.add_subplot(projection='3d')
    gtsam.utils.plot.plot_trajectory(fignum=0, values=optimized_estimates, title="Bundle Adjustment Trajectory")
    gtsam.utils.plot.set_axes_equal(0)
    ax.set_title(f"Left cameras and landmarks 2d trajectory for {len(cameras)} cameras.")
    ax.scatter(x_coordinates, y_coordinates,z_coordinates, s=50, c=colormap(10), alpha=.9)
    #Plot the trajectory in 2D
    landmarks_x = np.array([optimized_estimates.atPoint3(lm_sym)[0] for lm_sym in landmarks])
    landmarks_y = np.array([optimized_estimates.atPoint3(lm_sym)[1] for lm_sym in landmarks])
    landmarks_z = np.array([optimized_estimates.atPoint3(lm_sym)[2] for lm_sym in landmarks])
    ax.scatter(landmarks_x, landmarks_y, landmarks_z, s=50, c='blue')
    ax.view_init(vertical_axis='y')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    #plt.show()
    plt.savefig(PATH_TO_SAVE_3D_TRAJECTORY)

    # Step 7: Pick a Projection Factor and Compute Error


def q3():
    PATH_TO_SAVE_COMPARISON_TO_GT = "q3_compare_to_ground_truth"
    PATH_TO_SAVE_LOCALIZATION_ERROR = "q3_localization_error"
    PATH_TO_SAVE_2D_TRAJECTORY = "q3_2d_view_of_the_entire_scene"

    # Step 1: Select Keyframes
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    key_frames = choose_key_frames_by_elapsed_time(frameIds, 20)
    bundle_windows = get_bundle_windows(key_frames)

    # Step 2: Solve Every Bundle Window
    num_factor_in_bundles = []
    bundle_results = []
    for i, bundle_window in enumerate(bundle_windows):
        print(f"----------------Solving bundle # {i} from frame {bundle_window[0]} to frame {bundle_window[1]}------------------")
        bundle_graph, initial_estimates, landmarks, optimized_estimates = solve_one_bundle(track_db, bundle_window, debug=True)
        bundle_results.append((i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates))
        num_factor_in_bundles.append(bundle_graph.size())
        print(f"number of factors in graph is {num_factor_in_bundles[-1]}")
    print(f"FINISHED SOLVING BUNDLES!")
    print(f"Mean number of factors per graph is: {np.mean(num_factor_in_bundles)})")
    print(f"Min number of factors over graphs: {np.min(num_factor_in_bundles)})")
    print(f"Max number of factors over graphs: {np.max(num_factor_in_bundles)})")

    # Step 3: Check Error of Anchoring Factor of the Last Bundle
    i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates = bundle_results[-1]
    first_frameId_of_last_bundle = bundle_window[0]
    print(f"Printing optimized pose for first frame of the last bundle:")
    print(optimized_estimates.atPose3(gtsam.symbol(CAMERA,first_frameId_of_last_bundle)))
    anchoring_factor = bundle_graph.at(0)
    print(f"Anchoring Factor Error for this last bundle is: {anchoring_factor.error(optimized_estimates)}")

    # Step 4: Extracting Relative Poses Between Consecutive KeyFrames, and Also their Global Pose (relative to frame 0)
    optimized_relative_keyframes_poses = []
    optimized_global_keyframes_poses = []
    optimized_global_keyframes_poses.append(gtsam.Pose3())  # Initialize with first camera pose
    for bundle_res in bundle_results:
        i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates = bundle_res
        estimated_camera_position = optimized_estimates.atPose3(gtsam.symbol(CAMERA, bundle_window[1]))  # transforms from end of bundle to its beginning
        optimized_relative_keyframes_poses.append(estimated_camera_position)
        previous_global_pose = optimized_global_keyframes_poses[-1]  # transforms from beginning of bundle to global world
        current_global_pose = estimated_camera_position * previous_global_pose # transforms from end of bundle to global world
        optimized_global_keyframes_poses.append(current_global_pose)

    # Step 5: Compare Our Results With The Ground Truth Trajectory
    global_Rt_poses_in_numpy = np.array([pose.translation() for pose in optimized_global_keyframes_poses])
    gt_camera_positions = get_gt_trajectory()[np.array(key_frames)]
    plot_trajectories(camera_positions=global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions, path=PATH_TO_SAVE_COMPARISON_TO_GT)

    # Step 6: Presenting KeyFrame Localization Over Time
    plot_localization_error_over_time(key_frames, camera_positions=global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions, path=PATH_TO_SAVE_LOCALIZATION_ERROR)

    # Step 7: Presenting a View From Above (in 2d) of the Scene, With Keyframes and 3d Points
    #TODO I DIDN'T PRESENT YET A VIEW FROM OF THE SCENE WITH THE 3d POINTS (I ONLY ADDED THE CAMERA LOCATIONS). I'm not sure exactly what is required
    a = 5  # for you

if __name__ == "__main__":
    random.seed(6)
    s = gtsam.StereoCamera()
    track_db = TrackDatabase()
    deserialization_result = track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
    if deserialization_result == FAILURE:
        _, track_db = track_camera_for_many_images()
        track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)
    q3()
    exit()
    q1()

    q2()
