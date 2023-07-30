import gtsam
import numpy as np
from models.Constants import *
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from models.BundleAdjustment import bundle_adjustment, create_pose_graph, load_bundle_results
from utils.utils import track_camera_for_many_images, get_gt_trajectory
from utils.plotters import plot_trajectories, plot_localization_error_over_time
from models.LoopClosure import loop_closure, plot_pg_locations_before_and_after_lc,\
    plot_pg_locations_error_graph_before_and_after_lc, plot_pg_uncertainty_before_and_after_lc

class TrajectorySolver:

    def __init__(self, track_db):
        self.__matcher = Matcher()
        self.deserialization_result = None
        self.__track_db = track_db
        self._load_tracks_to_db()
        self.gt_trajectory = get_gt_trajectory()
        self.__predicted_trajectory = None

    def solve_trajectory(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_absolute_localization_error(self):
        raise NotImplementedError("Subclasses should implement this!")

    def compare_trajectory_to_gt(self):
        raise NotImplementedError("Subclasses should implement this!")

    def _load_tracks_to_db(self):
        self.deserialization_result = self.__track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)

        if self.deserialization_result == FAILURE:
            _, self.__track_db = track_camera_for_many_images()
            self.__track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)

    def get_track_db(self):
        return self.__track_db

    def get_deserialization_result(self):
        return self.deserialization_result

    def get_matcher(self):
        return self.__matcher


class PNP(TrajectorySolver):



    def __init__(self,track_db, force_recompute=False):
        super().__init__(track_db)
        self.force_recompute = force_recompute

    def compare_trajectory_to_gt(self):
        pass

    def solve_trajectory(self):
        if self.force_recompute or super().get_deserialization_result() != SUCCESS:
            camera_positions = track_camera_for_many_images()
            gt_camera_positions = get_gt_trajectory()
            plot_trajectories(camera_positions, gt_camera_positions)

    def get_absolute_localization_error(self):
        # Implement method to return PNP statistics here
        pass


class BundleAdjustment(TrajectorySolver):

    def __init__(self,track_db):
        super().__init__(track_db)
        self.bundle_results = None
        self.optimized_global_keyframes_poses = []
        self.bundle_3d_points = None
        self.optimized_relative_keyframes_poses = []
        self.global_3d_points = []
        self.key_frames = None

        self.__global_3d_points_numpy = None
        self.__global_Rt_poses_in_numpy = None

    def solve_trajectory(self):
        self.bundle_results = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
        self.key_frames = self.bundle_results[3]

    def compare_trajectory_to_gt(self):
        gt_camera_positions = get_gt_trajectory()[np.array(self.key_frames)]
        plot_trajectories(camera_positions=self.__global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions,
                          points_3d=self.__global_3d_points_numpy, path=PATH_TO_SAVE_COMPARISON_TO_GT_BUNDLE_ADJUSTMENT)

    def get_absolute_localization_error(self):
        try:
            plot_localization_error_over_time(self.key_frames, camera_positions=self.__global_Rt_poses_in_numpy,
                                          gt_camera_positions=get_gt_trajectory(), path=PATH_TO_SAVE_LOCALIZATION_ERROR_BUNDLE_ADJUSTMENT)
        except Exception:
            print("run solve trajectory first")


class LoopClosure(TrajectorySolver):


    def __init__(self,track_db):
        super().__init__(track_db)
        self.__pose_graph = None
        self.__our_trajectory = None
        self.__cur_pose_graph_estimates = None
        self.__successful_lc = None
        self.bundle_results = None
        self.optimized_global_keyframes_poses = []
        self.bundle_3d_points = None
        self.optimized_relative_keyframes_poses = []
        self.global_3d_points = []
        self.key_frames = None

        self.__global_3d_points_numpy = None
        self.__global_Rt_poses_in_numpy = None

    def solve_trajectory(self):
        # Implement Loop Closure algorithm here
        bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
        cond_matrices = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
        self.key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
        pose_graph, initial_estimates, landmarks = create_pose_graph(bundle_results,
                                                                     optimized_relative_keyframes_poses,
                                                                     optimized_global_keyframes_poses,
                                                                     cond_matrices)
        kf_to_covariance = {self.key_frames[i + 1]: cond_matrices[i] for i in range(len(cond_matrices))}
        cond_matrices = [cond_matrix * 10 for cond_matrix in cond_matrices]
        self.__our_trajectory = optimized_global_keyframes_poses
        self.__pose_graph, self.__cur_pose_graph_estimates, self.__successful_lc = loop_closure(pose_graph, self.key_frames,
                                                                           matcher=super().get_matcher(), cond_matrices=cond_matrices,
                                                                           mahalanobis_thresh=MAHALANOBIS_THRESH,
                                                                           pose_graph_initial_estimates=initial_estimates,
                                                                           draw_supporting_matches_flag=True,
                                                                           points_to_stop_by=True,
                                                                           compare_to_gt=True,
                                                                           show_localization_error=True,
                                                                           show_uncertainty=True)

    def get_absolute_localization_error(self):

        try:
            plot_pg_locations_error_graph_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)

        except Exception:
            print("run solve_trajectory() first")

    def compare_trajectory_to_gt(self):
        pass
