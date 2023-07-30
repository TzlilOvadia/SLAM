import models.BundleAdjustment
from utils.utils import track_camera_for_many_images
from models.LoopClosure import loop_closure
from models.Matcher import Matcher
from exercise_6 import *




if __name__ == "__main__":
    matcher = Matcher()

    random.seed(6)
    s = gtsam.StereoCamera()
    track_db = TrackDatabase()
    deserialization_result = track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
    if deserialization_result == FAILURE:
        _, track_db = track_camera_for_many_images()
        track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)
    bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
                                    cond_matrices = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
    key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
    pose_graph, initial_estimates, landmarks = models.BundleAdjustment.create_pose_graph(bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, cond_matrices)
    kf_to_covariance = {key_frames[i + 1]: cond_matrices[i] for i in range(len(cond_matrices))}
    cond_matrices = [cond_matrix * 10 for cond_matrix in cond_matrices]
    our_trajectory = optimized_global_keyframes_poses
    pose_graph, cur_pose_graph_estimates, successful_lc = loop_closure(pose_graph, key_frames,
                                                                       matcher=matcher, cond_matrices=cond_matrices,
                                                                       mahalanobis_thresh=MAHALANOBIS_THRESH,
                                                                       pose_graph_initial_estimates=initial_estimates,
                                                                       draw_supporting_matches_flag=True,
                                                                       points_to_stop_by=True,
                                                                       compare_to_gt=True,
                                                                       show_localization_error=True,
                                                                       show_uncertainty=True)

    # Printing the number of successful loop closures
    print(f"Overall, {len(successful_lc)} loop closures were detected.")
