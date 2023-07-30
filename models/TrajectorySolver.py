from models.Constants import *
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from models.BundleAdjustment import bundle_adjustment, create_pose_graph, load_bundle_results
from utils.utils import track_camera_for_many_images, get_gt_trajectory
from utils.plotters import plot_trajectories
from models.LoopClosure import loop_closure, plot_pg_locations_before_and_after_lc,\
    plot_pg_locations_error_graph_before_and_after_lc, plot_pg_uncertainty_before_and_after_lc


class TrajectorySolver:

    def __init__(self, track_db):
        self.__matcher = Matcher()
        self.__track_db = track_db
        self.load_tracks_to_db()
        self.gt_trajectory = get_gt_trajectory()
        self.__predicted_trajectory = None

    def solve_trajectory(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_stats(self):
        raise NotImplementedError("Subclasses should implement this!")

    def compare_trajectory_to_gt(self):
        raise NotImplementedError("Subclasses should implement this!")

    def load_tracks_to_db(self):
        deserialization_result = self.__track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
        if deserialization_result == FAILURE:
            _, self.__track_db = track_camera_for_many_images()
            self.__track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)

    def get_track_db(self):
        return self.__track_db

    def get_matcher(self):
        return self.__matcher


class PNP(TrajectorySolver):

    def compare_trajectory_to_gt(self):
        pass

    def __init__(self,track_db):
        super().__init__(track_db)

    def solve_trajectory(self):
        # Implement PNP algorithm here
        camera_positions = track_camera_for_many_images()
        gt_camera_positions = get_gt_trajectory()
        plot_trajectories(camera_positions, gt_camera_positions)

    def get_stats(self):
        # Implement method to return PNP statistics here
        pass


class BundleAdjustment(TrajectorySolver):

    def __init__(self,track_db):
        super().__init__(track_db)

    def solve_trajectory(self):
        bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
        cond_matrices = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)

    def get_stats(self):
        # Implement method to return Bundle Adjustment statistics here
        pass


class LoopClosure(TrajectorySolver):
    def __init__(self,track_db):
        super().__init__(track_db)
        self.__pose_graph = None
        self.__our_trajectory = None

    def solve_trajectory(self):


        # Implement Loop Closure algorithm here
        bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
        cond_matrices = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
        key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
        pose_graph, initial_estimates, landmarks = create_pose_graph(bundle_results,
                                                                     optimized_relative_keyframes_poses,
                                                                     optimized_global_keyframes_poses,
                                                                     cond_matrices)
        kf_to_covariance = {key_frames[i + 1]: cond_matrices[i] for i in range(len(cond_matrices))}
        cond_matrices = [cond_matrix * 10 for cond_matrix in cond_matrices]
        self.__our_trajectory = optimized_global_keyframes_poses
        pose_graph, cur_pose_graph_estimates, successful_lc = loop_closure(pose_graph, key_frames,
                                                                           matcher=super().get_matcher(), cond_matrices=cond_matrices,
                                                                           mahalanobis_thresh=MAHALANOBIS_THRESH,
                                                                           pose_graph_initial_estimates=initial_estimates,
                                                                           draw_supporting_matches_flag=True,
                                                                           points_to_stop_by=True,
                                                                           compare_to_gt=True,
                                                                           show_localization_error=True,
                                                                           show_uncertainty=True)

    def get_stats(self):
        # Implement method to return Loop Closure statistics here
        if compare_to_gt:
            plot_pg_locations_before_and_after_lc(pose_graph, cur_pose_graph_estimates, key_frames)
        if show_localization_error:
            plot_pg_locations_error_graph_before_and_after_lc(pose_graph, cur_pose_graph_estimates, key_frames)
        if show_uncertainty:
            plot_pg_uncertainty_before_and_after_lc(pose_graph, cur_pose_graph_estimates, key_frames)

