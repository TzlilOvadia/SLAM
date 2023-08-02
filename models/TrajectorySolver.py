import gtsam
import numpy as np
from models.Constants import *
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from models.BundleAdjustment import bundle_adjustment, create_pose_graph, load_bundle_results, get_translation_rotation_diff
from utils.utils import track_camera_for_many_images, get_gt_trajectory
from utils.plotters import plot_trajectories, plot_localization_error_over_time
from models.LoopClosure import loop_closure, plot_pg_locations_before_and_after_lc,\
    plot_pg_locations_error_graph_before_and_after_lc, plot_pg_uncertainty_before_and_after_lc, get_trajectory_from_graph


class TrajectorySolver:

    def __init__(self, track_db):
        self.__matcher = Matcher()
        self._final_estimated_trajectory = None
        self.deserialization_result = None
        self._track_db = track_db
        self._load_tracks_to_db()
        self._gt_trajectory = get_gt_trajectory()
        self._predicted_trajectory = None

    def solve_trajectory(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_absolute_localization_error(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_absolute_estimation_error(self):
        raise NotImplementedError("Subclasses should implement this!")

    def compare_trajectory_to_gt(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_final_estimated_trajectory(self):
        """
        Getter method for camera positions.

        Returns:
            list: List of camera positions.
        """
        return self._track_db.camera_positions

    def _load_tracks_to_db(self):
        """
        Method to load the tracks to the database.
        """
        self.deserialization_result = self._track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)

        if self.deserialization_result == FAILURE:
            self._final_estimated_trajectory, self._track_db = track_camera_for_many_images()
            self._track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)

    def get_track_db(self):
        """
        Getter method for the track database.

        Returns:
            TrackDatabase: The track database object.
        """
        return self._track_db

    def get_deserialization_result(self):
        """
        Getter method for the deserialization result.

        Returns:
            str: The result of the deserialization ('SUCCESS' or 'FAILURE').
        """
        return self.deserialization_result

    def get_matcher(self):
        """
        Getter method for the Matcher object.

        Returns:
            Matcher: The Matcher object.
        """
        return self.__matcher


class PNP(TrajectorySolver):

    def get_absolute_estimation_error(self):
        pass

    def __init__(self,track_db, force_recompute=False):
        super().__init__(track_db)
        self.force_recompute = force_recompute

    def compare_trajectory_to_gt(self):
        plot_trajectories(self._track_db.camera_positions, self._gt_trajectory)

    def solve_trajectory(self):
        if self.force_recompute or self.get_deserialization_result() != SUCCESS:
            _, self._track_db = track_camera_for_many_images()


    def get_absolute_localization_error(self):
        try:
            plot_localization_error_over_time(np.arange(len(list(self._track_db.get_frameIds()))), self._track_db.camera_positions, self._gt_trajectory,
                                              path="plots/pnp_localization_error_vs_key_frames", mode="PNP")

        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()




class BundleAdjustment(TrajectorySolver):

    def __init__(self,track_db):
        super().__init__(track_db)
        self.bundle_results = None
        self.optimized_relative_keyframes_poses = None
        self.optimized_global_keyframes_poses = None
        self.bundle_3d_points = None
        self.global_3d_points = []
        self.key_frames = None
        self.__global_3d_points_numpy = None
        self.__global_Rt_poses_in_numpy = None
        self._final_estimated_trajectory = None

    def solve_trajectory(self):
        self.bundle_results = load_bundle_results(PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS)
        bundle_windows = self.bundle_results[-2]
        self.key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
        self.__extract_trajectory_elements()
        bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
        cond_matrices = self.bundle_results
        key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
        _, initial_estimates, _ = create_pose_graph(bundle_results,
                                                                    optimized_relative_keyframes_poses,
                                                                    optimized_global_keyframes_poses,
                                                                    cond_matrices)
        self._final_estimated_trajectory = get_trajectory_from_graph(initial_estimates)
        self._gt_trajectory = get_gt_trajectory()[key_frames]

    def __extract_trajectory_elements(self):
        self.optimized_relative_keyframes_poses = self.bundle_results[2]
        self.optimized_global_keyframes_poses = self.bundle_results[1]
        optimized_relative_keyframes_poses = []
        optimized_global_keyframes_poses = []
        global_3d_points = []
        for bundle_res in self.bundle_results[0]:
            i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates = bundle_res
            estimated_camera_position = optimized_estimates.atPose3(
                gtsam.symbol(CAMERA, bundle_window[1]))  # transforms from end of bundle to its beginning
            optimized_relative_keyframes_poses.append(estimated_camera_position)
            previous_global_pose = self.optimized_global_keyframes_poses[
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
        try:
            plot_trajectories(camera_positions=self.__global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions,
                              points_3d=None, path=PATH_TO_SAVE_COMPARISON_TO_GT)
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_trajectories(camera_positions=self.__global_Rt_poses_in_numpy, gt_camera_positions=gt_camera_positions,
                              points_3d=None, path=PATH_TO_SAVE_COMPARISON_TO_GT)

    def get_rotation_error(self):

        pass
        # pred_trajectory = self.__global_Rt_poses_in_numpy
        # gt_trajectory = get_gt_trajectory()[self.key_frames]
        # for kf in self.key_frames:
        #     pred_pose = pred_trajectory[kf]
        #     gt_pose = gt_trajectory[kf]
        #     translation, rotation = get_translation_rotation_diff(pred_pose, gt_pose)
        #     print(translation)

    def get_absolute_localization_error(self):

        try:
            plot_localization_error_over_time(self.key_frames, camera_positions=self._final_estimated_trajectory,
                                          gt_camera_positions=self._gt_trajectory,
                                              path=PATH_TO_SAVE_LOCALIZATION_ERROR_BUNDLE_ADJUSTMENT,
                                              mode="Bundle Adjustment")
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_localization_error_over_time(self.key_frames, camera_positions=self.__global_Rt_poses_in_numpy,
                                              gt_camera_positions=self._gt_trajectory,
                                              path=PATH_TO_SAVE_LOCALIZATION_ERROR_BUNDLE_ADJUSTMENT,
                                              mode="Bundle Adjustment")


    def get_final_estimated_trajectory(self):
        return self._final_estimated_trajectory

class LoopClosure(TrajectorySolver):

    def __init__(self,track_db):
        super().__init__(track_db)
        self.__global_3d_points_numpy = None
        self.__global_Rt_poses_in_numpy = None
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
                                                                           points_to_stop_by=True
                                                                           )

    def compare_trajectory_to_gt(self):
        try:
            plot_pg_locations_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_pg_locations_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)

    def get_absolute_localization_error(self):
        try:
            plot_pg_locations_error_graph_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_pg_locations_error_graph_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)

    def get_rotation_error(self):
        pass
        # pred_trajectory = self.get_final_estimated_trajectory()
        # gt_trajectory = get_gt_trajectory()[self.key_frames]
        # for kf in self.key_frames:
        #     pred_pose = pred_trajectory[kf]
        #     gt_pose = gt_trajectory[kf]
        #     translation, rotation = get_translation_rotation_diff(pred_pose, gt_pose)
        #     print(translation)

    def show_uncertainty(self):
        try:
            plot_pg_uncertainty_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_pg_uncertainty_before_and_after_lc(self.__pose_graph, self.__cur_pose_graph_estimates)

    def get_final_estimated_trajectory(self):
        return get_trajectory_from_graph(self.__cur_pose_graph_estimates)