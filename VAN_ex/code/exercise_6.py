import random

import utils.utils
from models.BundleAdjustment import get_bundle_windows, key_frames_by_percentile, solve_one_bundle, load_bundle_results, \
    get_conditional_covariance_matrix_between_keyframes, get_relative_pose_between_frames, create_pose_graph
from models.TrackDatabase import TrackDatabase
from utils import utils
from models.Constants import *
import gtsam
import utils.plot as plot_helper



K, M1, M2 = utils.read_cameras()
GTSAM_K = utils.get_gtsam_calib_mat(K, M2)


def q1():
    PATH_FOR_TRAJECTORY_WITH_COVARIANCES = "ex6_q1_trajectory"
    # PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker_3"
    # Step 1: Select Keyframes
    track_db = TrackDatabase(PATH_TO_SAVE_TRACKER_FILE)
    frameIds = track_db.get_frameIds()
    key_frames = key_frames_by_percentile(list(frameIds.keys()), .85, track_db)
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



def q2(force_recompute=False, debug=True):
    # PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker_2"
    # Step 1: get bundle adjustment results
    bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, cond_matrices = load_bundle_results(path=PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS, force_recompute=force_recompute, debug=debug)
    key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
    print("Creating Pose Graph...")
    pose_graph, initial_estimates, landmarks = create_pose_graph(bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, cond_matrices)
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



if __name__ == '__main__':
    # import exercise_4

    random.seed(6)

    # load tracking data
    s = gtsam.StereoCamera()
    track_db = TrackDatabase()
    # cp, track_db = exercise_4.track_camera_for_many_images()
    # track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)
    # exit()
    deserialization_result = track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
    if deserialization_result == FAILURE:
        _, track_db = utils.utils.track_camera_for_many_images()
        track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)
    # solve exercise questions
    q1()
    q2(force_recompute=False, debug=True)
    exit()



    # TODO: A) make the visuzlization work (plot trajectory).
    # TODO: B) make the covariance visualization in particular work (it will probably be too large covariances).
    # TODO C) verify that we convert the contiditional covariance matrices to GTSAM correctly.