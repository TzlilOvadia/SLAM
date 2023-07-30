from models.BundleAdjustment import get_bundle_windows, select_keyframes_by_track_max_distance, \
    key_frames_by_percentile, triangulate_and_project, compute_factor_error, solve_one_bundle
from utils import utils
from utils.plotters import plot_trajectories, plot_localization_error_over_time, plot_projections_on_images, \
    plot_2d_cameras_and_points
from utils.utils import *
from matplotlib import pyplot as plt
from models.Constants import *
import gtsam
from gtsam.utils import plot

K, M1, M2 = utils.read_cameras()
GTSAM_K = utils.get_gtsam_calib_mat(K, M2)



def q1():
    PATH_TO_SAVE_R_ERROR_OVER_TRACK_FRAMES = "q1_r_error_per_frame"
    PATH_TO_SAVE_FACTOR_ERROR_OVER_TRACK_FRAMES = "q1_factor_error_per_frame"
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    trackId, track = track_db.get_random_track_of_length(10)
    reprojection_errors, factors, values = triangulate_and_project(track, track_db)
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

    plt.figure()
    plt.plot(reprojection_errors, factor_errors)
    plt.xlabel("reprojection error")
    plt.ylabel('factor error')
    plt.title("Factor error as function of ReProjection Error")
    plt.savefig(PATH_TO_SAVE_FACTOR_ERROR_OVER_TRACK_FRAMES + "as_func_of_reproj_error")

def q2():
    PATH_TO_SAVE_3D_TRAJECTORY = "q2_3d_trajectory_after_optimization"
    PATH_TO_SAVE_3D_INITIAL_TRAJECTORY = "q2_3d_trajectory_before_optimization"
    PATH_TO_SAVE_2D_TRAJECTORY = "q2_2d_trajectory_after_optimization" # TODO ADD THIS REQUIRED VISUALIZATION
    # Step 1: Select Keyframes
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    # key_frames = choose_key_frames_by_elapsed_time(frameIds, 10)
    key_frames= select_keyframes_by_track_max_distance(frameIds,track_db)
    # select_keyframes_distance(frameIds.keys())
    bundle_windows = get_bundle_windows(key_frames)

    # Step 2: Define Bundle Optimization
    first_bundle = bundle_windows[0]
    bundle_graph, initial_estimates, landmarks, optimized_estimates, bundle_covariance = solve_one_bundle(track_db, first_bundle, debug=True)

    landmarks_x = np.array([optimized_estimates.atPoint3(lm_sym)[0] for lm_sym in landmarks])
    landmarks_y = np.array([optimized_estimates.atPoint3(lm_sym)[1] for lm_sym in landmarks])
    landmarks_z = np.array([optimized_estimates.atPoint3(lm_sym)[2] for lm_sym in landmarks])

    # Plot the Resulting Positions

    # Plotting the optimized trajectory as a 3D graph
    bundle_starts_in_frame_id, bundle_ends_in_frame_id =  first_bundle
    optimized_cameras = np.array([optimized_estimates.atPose3(gtsam.symbol(CAMERA,frameId)) for frameId in range(bundle_starts_in_frame_id, bundle_ends_in_frame_id+1)])  # List of gtsam.Pose3 objects representing poses
    x_coordinates = [pose.x() for pose in optimized_cameras]
    y_coordinates = [pose.y() for pose in optimized_cameras]
    z_coordinates = [pose.z() for pose in optimized_cameras]
    fig = plt.figure(num=0)
    ax=fig.add_subplot(projection='3d')
    gtsam.utils.plot.plot_trajectory(fignum=0, values=optimized_estimates, title="Bundle Adjustment Trajectory")
    gtsam.utils.plot.set_axes_equal(0)
    ax.set_title(f"Left cameras 3d optimized trajectory for {len(optimized_cameras)} cameras.")
    ax.scatter(x_coordinates, y_coordinates,z_coordinates, s=50)
    ax.scatter(landmarks_x, landmarks_y, landmarks_z, s=20, c='red', label='landmarks')
    ax.view_init(vertical_axis='y')
    #plt.show()
    plt.savefig(PATH_TO_SAVE_3D_TRAJECTORY)


    # Plotting the initial estimates trajectory as a 3D graph
    cameras = np.array([initial_estimates.atPose3(gtsam.symbol(CAMERA,frameId)) for frameId in range(bundle_starts_in_frame_id, bundle_ends_in_frame_id+1)])
    x_coordinates = [pose.x() for pose in cameras]
    y_coordinates = [pose.y() for pose in cameras]
    z_coordinates = [pose.z() for pose in cameras]
    fig = plt.figure(num=0)
    ax = fig.add_subplot(projection='3d')
    gtsam.utils.plot.plot_trajectory(fignum=0, values=initial_estimates, title="Bundle Adjustment Trajectory")
    gtsam.utils.plot.set_axes_equal(0)
    ax.set_title(f"Left cameras 3d initial  trajectory for {len(cameras)} cameras.")
    ax.scatter(x_coordinates, y_coordinates, z_coordinates, s=50)
    ax.view_init(vertical_axis='y')
    #plt.show()
    plt.savefig(PATH_TO_SAVE_3D_INITIAL_TRAJECTORY)

    plot_2d_cameras_and_points(x_coordinates, z_coordinates, landmarks_x, landmarks_z, PATH_TO_SAVE_2D_TRAJECTORY)

    # Step 7: Pick a Projection Factor and Compute Error
    print(f"Choosing some projection factor from the bundle graph...")
    some_factor = bundle_graph.at(20)
    print(f"Printing the factor's error over the initial estimates: {some_factor.error(initial_estimates)}")
    print(f"Printing the factor's error over the optimized estimates: {some_factor.error(optimized_estimates)}")
    initial_c_pose = initial_estimates.atPose3(some_factor.keys()[0])
    initial_q = initial_estimates.atPoint3(some_factor.keys()[1])
    optimized_q = optimized_estimates.atPoint3(some_factor.keys()[1])
    stereo_camera = gtsam.StereoCamera(initial_c_pose, GTSAM_K)
    stereo_point_2d = stereo_camera.project(initial_q)
    measured_point_2d = some_factor.measured()
    optimized_point_2d = stereo_camera.project(optimized_q)
    left_image, right_image = read_images(9)
    plot_projections_on_images(left_image, right_image, measured_point_2d, stereo_point_2d, optimized_point_2d)

    # all_points = gtsam.utilities.extractPoint3(optimized_estimates)
    # all_poses = gtsam.utilities.extractPose3(optimized_estimates).reshape(-1, 4, 3).transpose(0, 2, 1)
    # camera_positions = np.array([pose[:,-1] for pose in all_poses])
    # plot_trajectory_and_points(camera_positions,all_points)

def q3():
    PATH_TO_SAVE_COMPARISON_TO_GT = "q3_compare_to_ground_truth"
    PATH_TO_SAVE_LOCALIZATION_ERROR = "q3_localization_error"
    PATH_TO_SAVE_2D_TRAJECTORY = "q3_2d_view_of_the_entire_scene"

    # Step 1: Select Keyframes
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    key_frames = key_frames_by_percentile(list(frameIds.keys()), .87, track_db)
    prev = key_frames[0]
    for frame in key_frames[1:]:
        if frame-prev < 5:
            print(f"window size is {frame-prev}")
        prev = frame
    bundle_windows = get_bundle_windows(key_frames)
    # Step 2: Solve Every Bundle Window
    num_factor_in_bundles = []
    bundle_results = []
    for i, bundle_window in enumerate(bundle_windows):
        print(f"----------------Solving bundle # {i} from frame {bundle_window[0]} to frame {bundle_window[1]}------------------")
        bundle_graph, initial_estimates, landmarks, optimized_estimates, bundle_covariance = solve_one_bundle(track_db, bundle_window, debug=False)
        bundle_results.append((i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates))
        num_factor_in_bundles.append(bundle_graph.size())
        print(f"number of factors in graph is {num_factor_in_bundles[-1]}")
        initial_error = bundle_graph.error(initial_estimates)
        optimized_error = bundle_graph.error(optimized_estimates)
        print(f"initial graph error was {initial_error}, and after optimization the error is {optimized_error}")
        fig = plt.figure(num=0)
        ax = fig.add_subplot(projection='3d')
        gtsam.utils.plot.plot_trajectory(fignum=0, values=optimized_estimates, title="Bundle Adjustment Trajectory")
        gtsam.utils.plot.set_axes_equal(0)
        ax.view_init(vertical_axis='y')
        plt.show()
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
    _, bundle_window, _, _, _, optimized_estimates = bundle_results[0]
    estimated_camera_position = optimized_estimates.atPose3(gtsam.symbol(CAMERA, bundle_window[0]))
    optimized_global_keyframes_poses.append(estimated_camera_position)
    #optimized_global_keyframes_poses.append(gtsam.Pose3())  # Initialize with first camera pose
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

    # Step 5: Compare Our Results With The Ground Truth Trajectory
    global_3d_points_numpy = np.array(global_3d_points)
    global_Rt_poses_in_numpy = np.array([pose.translation() for pose in optimized_global_keyframes_poses])
    gt_camera_positions = get_gt_trajectory()[np.array(key_frames)]
    plot_trajectories(camera_positions=global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions, path=PATH_TO_SAVE_COMPARISON_TO_GT)
    plt.savefig(PATH_TO_SAVE_2D_TRAJECTORY)

    # Step 6: Presenting KeyFrame Localization Over Time
    plot_localization_error_over_time(key_frames, camera_positions=global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions, path=PATH_TO_SAVE_LOCALIZATION_ERROR)
    plt.savefig(PATH_TO_SAVE_LOCALIZATION_ERROR)
    # Step 7: Presenting a View From Above (in 2d) of the Scene, With Keyframes and 3d Points
    plot_trajectories(camera_positions=global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions, points_3d=global_3d_points_numpy, path=PATH_TO_SAVE_COMPARISON_TO_GT)


def bundle_adjustment(path_to_serialize=None, debug=False, plot_results=None):
    PATH_TO_SAVE_COMPARISON_TO_GT = "q3_compare_to_ground_truth"
    PATH_TO_SAVE_LOCALIZATION_ERROR = "q3_localization_error"
    PATH_TO_SAVE_2D_TRAJECTORY = "q3_2d_view_of_the_entire_scene"

    # Step 1: Select Keyframes
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    key_frames = key_frames_by_percentile(list(frameIds.keys()), .87, track_db)

    bundle_windows = get_bundle_windows(key_frames)
    # Step 2: Solve Every Bundle Window
    num_factor_in_bundles = []
    bundle_results = []
    for i, bundle_window in enumerate(bundle_windows):

        bundle_graph, initial_estimates, landmarks, optimized_estimates, marginals = solve_one_bundle(track_db, bundle_window, debug=False)
        bundle_results.append((i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates, marginals))
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
    _, bundle_window, _, _, _, optimized_estimates, marginals = bundle_results[0]
    estimated_camera_position = optimized_estimates.atPose3(gtsam.symbol(CAMERA, bundle_window[0]))
    optimized_global_keyframes_poses.append(estimated_camera_position)
    #optimized_global_keyframes_poses.append(gtsam.Pose3())  # Initialize with first camera pose
    global_3d_points = []
    for bundle_res in bundle_results:
        i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates, marginals = bundle_res
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

    if path_to_serialize:
        import pickle
        bundle_adjustment_results_dict = {"bundle_results": bundle_results,
                                          "optimized_relative_keyframes_poses": optimized_relative_keyframes_poses,
                                          "optimized_global_keyframes_poses": optimized_global_keyframes_poses,
                                          "bundle_windows": bundle_windows}
        with open(path_to_serialize, 'wb') as f:
            pickle.dump(bundle_adjustment_results_dict, f)
    return bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows


if __name__ == "__main__":
    random.seed(6)
    s = gtsam.StereoCamera()
    track_db = TrackDatabase()
    deserialization_result = track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
    if deserialization_result == FAILURE:
        _, track_db = utils.utils.track_camera_for_many_images()
        track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)

    q3()
    #q1()
    #q2()
    # q2()
    # exit()

