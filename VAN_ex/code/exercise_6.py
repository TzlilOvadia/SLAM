import random
import cv2
import numpy as np
from tqdm import tqdm
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


PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker"
PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS = "../../models/bundle_adjustment_results"
K, M1, M2 = utils.read_cameras()
GTSAM_K = utils.get_gtsam_calib_mat(K, M2)


def load_bundle_results(path=None, force_recompute=False, debug=True):
    if path is None:
        path = PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS
    if force_recompute:
        print("Recomputing Bundle Adjustment ...")
        return bundle_adjustment(path_to_serialize=path, debug=debug, plot_results=True)
    try:
        with open(path, 'rb') as f:
            bundle_adjustment_results = pickle.load(f)
            bundle_results = bundle_adjustment_results["bundle_results"]
            optimized_relative_keyframes_poses = bundle_adjustment_results["optimized_relative_keyframes_poses"]
            optimized_global_keyframes_poses = bundle_adjustment_results["optimized_global_keyframes_poses"]
            bundle_windows = bundle_adjustment_results["bundle_windows"]
            cond_matrices = bundle_adjustment_results["cond_matrices"]
            print(f"Found File At {path}, loaded bundle adjustment results...")
    except Exception:
        print(f"No File Exists In {path}")
        print("Recomputing Bundle Adjustment ...")
        return bundle_adjustment(path_to_serialize=path, debug=debug, plot_results=True)
    return bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, cond_matrices


def q1():
    PATH_FOR_TRAJECTORY_WITH_COVARIANCES = "ex6_q1_trajectory"
    PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker"
    # Step 1: Select Keyframes
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    key_frames = criteria(list(frameIds.keys()), .85, track_db)
    bundle_windows = get_bundle_windows(key_frames)

    # Step 2: Define Bundle Optimization
    first_window = bundle_windows[0]

    bundle_graph, initial_estimates, landmarks, optimized_estimates, marginals = solve_one_bundle(track_db, first_window,
                                                                                       debug=True)

    # Plotting the optimized trajectory as a 3D graph
    cond_cov_mat = get_conditional_covariance_matrix_between_keyframes(first_window, marginals)
    relative_pose = get_relative_pose_between_frames(first_window[0], first_window[1], optimized_estimates)

    plot_helper.plot_trajectory(1, optimized_estimates, marginals=marginals, scale=1,
                                title="First window's covariance poses", save_file=PATH_FOR_TRAJECTORY_WITH_COVARIANCES)
    #ax.view_init(vertical_axis='y')
    #plt.show()
    #plt.savefig(PATH_FOR_TRAJECTORY_WITH_COVARIANCES)
    print(f"Relative covariance between pose in frames {first_window[0]} and {first_window[1]}:\n", cond_cov_mat, "\n")
    print("last key frame relative pose :\n", relative_pose)


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
    # Retrieve optimized poses for cameras ci and ck.
    camera_a = estimates.atPose3(gtsam.symbol(CAMERA, frame_a))
    camera_b = estimates.atPose3(gtsam.symbol(CAMERA, frame_b))

    # Calculate the relative pose from camera ci to camera ck.
    relative_pose = camera_a.between(camera_b)
    return relative_pose


def create_pose_graph(bundle_results, rel_poses_lst, optimized_global_keyframes_poses, bundle_windows, rel_cov_mat_lst):
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


def q2(force_recompute=False, debug=True):
    PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker"
    # Step 1: get bundle adjustment results
    bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, cond_matrices = load_bundle_results(path=PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS, force_recompute=force_recompute, debug=debug)
    key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
    print("Creating Pose Graph...")
    pose_graph, initial_estimates, landmarks = create_pose_graph(bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, cond_matrices)
    print("Optimizing graph...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(pose_graph, initial_estimates)
    optimized_estimates = optimizer.optimize()
    print("Graph error BEFORE optimization: ", pose_graph.error(initial_estimates))
    print("Graph error AFTER optimization: ", pose_graph.error(optimized_estimates))


    # Plot initial estimate trajectory
    plot_helper.plot_trajectory(1, initial_estimates, scale=1, title="q2_initial_estimates_trajectory",
                                     save_file="ex6_q2_initial_estimates_trajectory")

    # Plot optimized trajectory without covariance
    plot_helper.plot_trajectory(1, optimized_estimates, scale=1, title="q2_optimized_estimates_trajectory",
                                     save_file="ex6_q2_optimized_estimates_trajectory")

    # Optimized trajectory with covariance
    marginals = gtsam.Marginals(pose_graph, optimized_estimates)
    plot_helper.plot_trajectory(1, optimized_estimates, marginals=marginals,
                         title="q2_optimized_estimates_trajectory_with_cov", scale=1,
                         save_file="ex6_q2_optimized_estimates_trajectory_with_cov")



def bundle_adjustment(path_to_serialize=None, debug=False, plot_results=None, track_db = None):
    """
    This function solves bundle adjustment for many bundles independently, and calculates the resulting trajectory.
    :param path_to_serialize: path in which the bundle results will be saved at.
    :param debug: if True execution will include printouts.
    :param plot_results: if True a plot of the trajectory in comparison to the Ground Truth will be presented.
    :param track_db: The tracking data.
    :return: Results of bundle adjustment
    """
    PATH_TO_SAVE_COMPARISON_TO_GT = "q3_compare_to_ground_truth"
    PATH_TO_SAVE_LOCALIZATION_ERROR = "q3_localization_error"
    PATH_TO_SAVE_2D_TRAJECTORY = "q3_2d_view_of_the_entire_scene"

    # Step 1: Select Keyframes
    if track_db is None:
        track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    key_frames = criteria(list(frameIds.keys()), .85, track_db)

    bundle_windows = get_bundle_windows(key_frames)
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
                                          "bundle_windows": bundle_windows,
                                          "cond_matrices": cond_matrices}
        with open(path_to_serialize, 'wb') as f:
            pickle.dump(bundle_adjustment_results_dict, f)
    return bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, cond_matrices


if __name__ == '__main__':
    import exercise_4, pickle
    random.seed(6)

    # load tracking data
    s = gtsam.StereoCamera()
    track_db = TrackDatabase()
    deserialization_result = track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
    if deserialization_result == FAILURE:
        _, track_db = exercise_4.track_camera_for_many_images()
        track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)

    # solve exercise questions
    q1()
    q2(force_recompute=False, debug=True)
    exit()



    # TODO: A) make the visuzlization work (plot trajectory).
    # TODO: B) make the covariance visualization in particular work (it will probably be too large covariances).
    # TODO C) verify that we convert the contiditional covariance matrices to GTSAM correctly.