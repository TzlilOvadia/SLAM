import gtsam
import numpy as np
from models.Constants import *
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from models.BundleAdjustment import bundle_adjustment, create_pose_graph, load_bundle_results
from utils.utils import track_camera_for_many_images, get_gt_trajectory
from utils.plotters import plot_trajectories, plot_localization_error_over_time
from models.LoopClosure import loop_closure, plot_pg_locations_before_and_after_lc,\
    plot_pg_locations_error_graph_before_and_after_lc, plot_pg_uncertainty_before_and_after_lc, get_trajectory_from_graph


class TrajectorySolver:

    def __init__(self, track_db):
        self.__matcher = Matcher()
        self._camera_positions = None
        self.deserialization_result = None
        self._track_db = track_db
        self._load_tracks_to_db()
        self.gt_trajectory = get_gt_trajectory()
        self._predicted_trajectory = None

    def solve_trajectory(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_absolute_localization_error(self):
        raise NotImplementedError("Subclasses should implement this!")

    def compare_trajectory_to_gt(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_camera_positions(self):
        return self._camera_positions

    def _load_tracks_to_db(self):
        self.deserialization_result = self._track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)

        if self.deserialization_result == FAILURE:
            self._camera_positions, self._track_db = track_camera_for_many_images()
            self._track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)

    def get_track_db(self):
        return self._track_db

    def get_deserialization_result(self):
        return self.deserialization_result

    def get_matcher(self):
        return self.__matcher


class PNP(TrajectorySolver):

    def __init__(self,track_db, force_recompute=False):
        super().__init__(track_db)
        self.force_recompute = force_recompute

    def compare_trajectory_to_gt(self):
        plot_trajectories(self._track_db.camera_positions, self.gt_trajectory)

    def solve_trajectory(self):
        if self.force_recompute or self.get_deserialization_result() != SUCCESS:
            _, self._track_db = track_camera_for_many_images()


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
        self._camera_positions = None

    def solve_trajectory(self):
        self.bundle_results = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
        self.__extract_trajectory_elements()

    def __extract_trajectory_elements(self):
        optimized_relative_keyframes_poses = self.bundle_results[2]
        optimized_global_keyframes_poses = self.bundle_results[1]
        global_3d_points = []
        for bundle_res in self.bundle_results[0]:
            i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates = bundle_res
            estimated_camera_position = optimized_estimates.atPose3(
                gtsam.symbol(CAMERA, bundle_window[1]))  # transforms from end of bundle to its beginning
            optimized_relative_keyframes_poses.append(estimated_camera_position)
            previous_global_pose = optimized_global_keyframes_poses[
                -1]  # transforms from beginning of bundle to global world
            current_global_pose = previous_global_pose * estimated_camera_position  # transforms from end of bundle to global world
            bundle_3d_points = gtsam.utilities.extractPoint3(optimized_estimates)
            for point in bundle_3d_points:
                global_point = previous_global_pose.transformFrom(gtsam.Point3(point))
                global_3d_points.append(global_point)
            optimized_global_keyframes_poses.append(current_global_pose)
        self.__global_3d_points_numpy = np.array(global_3d_points)
        self.__global_Rt_poses_in_numpy = np.array([pose.translation() for pose in optimized_global_keyframes_poses])

    def compare_trajectory_to_gt(self):
        gt_camera_positions = get_gt_trajectory()
        plot_trajectories(camera_positions=self.__global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions,
                          points_3d=global_3d_points_numpy, path=PATH_TO_SAVE_COMPARISON_TO_GT)


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
        # kf_to_covariance = {self.key_frames[i + 1]: cond_matrices[i] for i in range(len(cond_matrices))}
        cond_matrices = [cond_matrix * 10 for cond_matrix in cond_matrices]
        self.__our_trajectory = optimized_global_keyframes_poses
        self.__pose_graph, self.__cur_pose_graph_estimates, self.__successful_lc = loop_closure(pose_graph, self.key_frames,
                                                                           matcher=super().get_matcher(), cond_matrices=cond_matrices,
                                                                           mahalanobis_thresh=MAHALANOBIS_THRESH,
                                                                           pose_graph_initial_estimates=initial_estimates,
                                                                           draw_supporting_matches_flag=True,
                                                                           points_to_stop_by=True
                                                                           )

    def compare_trajectory_to_gt(self):
        plot_pg_locations_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)

    def get_absolute_localization_error(self):
        plot_pg_locations_error_graph_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)

    def show_uncertainty(self):
        plot_pg_uncertainty_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)