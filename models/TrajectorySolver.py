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

    def __init__(self, path_to_track_db=PATH_TO_SAVE_TRACKER_FILE):
        self.__matcher = Matcher()
        self._final_estimated_trajectory = None
        self.deserialization_result = None
        self._track_db = TrackDatabase()
        self._load_tracks_to_db(path=path_to_track_db)
        self._gt_trajectory = get_gt_trajectory()
        self._predicted_trajectory = None


    def get_rotation_error(self):
        raise NotImplementedError("Subclasses should implement this!")

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

    def _load_tracks_to_db(self, path=PATH_TO_SAVE_TRACKER_FILE):
        """
        Method to load the tracks to the database.
        """
        self.deserialization_result = self._track_db.deserialize(path)

        if self.deserialization_result == FAILURE:
            self._final_estimated_trajectory, self._track_db = track_camera_for_many_images()
            self._track_db.serialize(path)

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

    def get_trajectory_from_poses(self, poses):
        trajectory = np.array([pose.translation() for pose in poses])
        return trajectory


class PNP(TrajectorySolver):

    def get_absolute_estimation_error(self):
        pass

    def __init__(self, force_recompute=False, path_to_save_track_db=PATH_TO_SAVE_TRACKER_FILE):
        super().__init__(path_to_track_db=path_to_save_track_db)
        self.force_recompute = force_recompute

    def compare_trajectory_to_gt(self, path_suffix=""):
        plot_trajectories(self._track_db.camera_positions, self._gt_trajectory, path=PATH_TO_SAVE_COMPARISON_TO_GT_PNP + path_suffix,
                          suffix="(PNP)")

    def solve_trajectory(self):
        if self.force_recompute or self.get_deserialization_result() != SUCCESS:
            _, self._track_db = track_camera_for_many_images()


    def get_absolute_localization_error(self, path_suffix=""):
        try:
            plot_localization_error_over_time(np.arange(len(list(self._track_db.get_frameIds()))), self._track_db.camera_positions, self._gt_trajectory,
                                              path="plots/pnp_localization_error_vs_key_frames" + path_suffix, mode="PNP")

        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_localization_error_over_time(np.arange(len(list(self._track_db.get_frameIds()))), self._track_db.camera_positions, self._gt_trajectory,
                                              path="plots/pnp_localization_error_vs_key_frames" + path_suffix, mode="PNP")


class BundleAdjustment(TrajectorySolver):

    def __init__(self, force_recompute=False, path_to_save_track_db=PATH_TO_SAVE_TRACKER_FILE,
                 path_to_save_ba=PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS):
        super().__init__(path_to_track_db=path_to_save_track_db)
        self.bundle_results = None
        self.optimized_relative_keyframes_poses = None
        self.optimized_global_keyframes_poses = None
        self.bundle_3d_points = None
        self.global_3d_points = []
        self.key_frames = None
        self._final_estimated_trajectory = None
        self.force_recompute = force_recompute
        self.path_to_save_db = path_to_save_track_db
        self.path_to_save_ba = path_to_save_ba

    def solve_trajectory(self):
        self.bundle_results = load_bundle_results(path=self.path_to_save_ba, force_recompute=self.force_recompute, track_db_path=self.path_to_save_db)
        bundle_windows = self.bundle_results[-2]
        self.key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
        bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
        cond_matrices = self.bundle_results
        key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
        _, initial_estimates, _ = create_pose_graph(bundle_results,
                                                                    optimized_relative_keyframes_poses,
                                                                    optimized_global_keyframes_poses,
                                                                    cond_matrices)
        self._final_estimated_trajectory = get_trajectory_from_graph(initial_estimates)
        self._gt_trajectory = get_gt_trajectory()[key_frames]

    def compare_trajectory_to_gt(self, path_suffix=""):
        gt_camera_positions = self._gt_trajectory
        try:
            plot_trajectories(camera_positions=self._final_estimated_trajectory, gt_camera_positions=gt_camera_positions,
                              points_3d=None, path=PATH_TO_SAVE_COMPARISON_TO_GT_BUNDLE_ADJUSTMENT + path_suffix, suffix="(BUNDLE ADJUSTMENT)")
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_trajectories(camera_positions=self._final_estimated_trajectory, gt_camera_positions=gt_camera_positions,
                              points_3d=None, path=PATH_TO_SAVE_COMPARISON_TO_GT_BUNDLE_ADJUSTMENT + path_suffix, suffix="(BUNDLE ADJUSTMENT)")

    def get_rotation_error(self):

        pass
        # pred_trajectory = self.__global_Rt_poses_in_numpy
        # gt_trajectory = get_gt_trajectory()[self.key_frames]
        # for kf in self.key_frames:
        #     pred_pose = pred_trajectory[kf]
        #     gt_pose = gt_trajectory[kf]
        #     translation, rotation = get_translation_rotation_diff(pred_pose, gt_pose)
        #     print(translation)

    def get_absolute_localization_error(self, path_suffix=""):

        try:
            plot_localization_error_over_time(self.key_frames, camera_positions=self._final_estimated_trajectory,
                                          gt_camera_positions=self._gt_trajectory,
                                              path=PATH_TO_SAVE_LOCALIZATION_ERROR_BUNDLE_ADJUSTMENT + path_suffix,
                                              mode="Bundle Adjustment")
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_localization_error_over_time(self.key_frames, camera_positions=self._final_estimated_trajectory,
                                              gt_camera_positions=self._gt_trajectory,
                                              path=PATH_TO_SAVE_LOCALIZATION_ERROR_BUNDLE_ADJUSTMENT + path_suffix,
                                              mode="Bundle Adjustment")

    def get_final_estimated_trajectory(self):
        return self._final_estimated_trajectory


class LoopClosure(TrajectorySolver):

    def __init__(self,  path_to_save_track_db=PATH_TO_SAVE_TRACKER_FILE,
                 path_to_save_ba=PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS,
                 path_to_save_lc=PATH_TO_SAVE_LOOP_CLOSURE_RESULTS,
                 force_recompute=False):
        super().__init__(path_to_track_db=path_to_save_track_db)
        self._global_3d_points_numpy = None
        self._global_Rt_poses_in_numpy = None
        self._pose_graph = None
        self._initial = None
        self._cur_pose_graph_estimates = None
        self._successful_lc = None
        self.bundle_results = None
        self.optimized_global_keyframes_poses = []
        self.bundle_3d_points = None
        self.optimized_relative_keyframes_poses = []
        self.global_3d_points = []
        self.key_frames = None
        self._final_estimated_trajectory = None
        self._good_mahalanobis_candidates = None
        self._initial_trajectory = None
        self.path_to_save_db = path_to_save_track_db
        self.path_to_save_ba = path_to_save_ba
        self.path_to_save_lc = path_to_save_lc
        self.force_recompute = force_recompute
        if self.force_recompute:
            print("Forcing recompute on loop closure even if file is found...")


    def solve_trajectory(self):
        # Implement Loop Closure algorithm here
        deserialization_results = LoopClosure.deserialize(self, self.path_to_save_lc)
        if deserialization_results == FAILURE or self.force_recompute:
            bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
            cond_matrices = load_bundle_results(path=self.path_to_save_ba, track_db_path=self.path_to_save_db)
            self.key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
            pose_graph, initial_estimates, landmarks = create_pose_graph(bundle_results,
                                                                         optimized_relative_keyframes_poses,
                                                                         optimized_global_keyframes_poses,
                                                                         cond_matrices)
            kf_to_covariance = {self.key_frames[i + 1]: cond_matrices[i] for i in range(len(cond_matrices))}
            cond_matrices = [cond_matrix * 10 for cond_matrix in cond_matrices]
            self._initial_trajectory = self.get_trajectory_from_poses(optimized_global_keyframes_poses)
            self._pose_graph, self._cur_pose_graph_estimates, self._successful_lc, good_ms = loop_closure(pose_graph, self.key_frames,
                                                                                                          matcher=super().get_matcher(), cond_matrices=cond_matrices,
                                                                                                          mahalanobis_thresh=MAHALANOBIS_THRESH,
                                                                                                          pose_graph_initial_estimates=initial_estimates,
                                                                                                          draw_supporting_matches_flag=False,
                                                                                                          points_to_stop_by=False
                                                                                                          )
            self._good_mahalanobis_candidates = [(c[1], c[2]) for c in good_ms]
            self._final_estimated_trajectory = self.get_final_estimated_trajectory()
            self._gt_trajectory = get_gt_trajectory()[self.key_frames]
            self.serialize(self.path_to_save_lc)

    def compare_trajectory_to_gt(self, path_suffix=""):
        gt_camera_positions =  self._gt_trajectory
        try:
            plot_trajectories(camera_positions=self._final_estimated_trajectory, gt_camera_positions=gt_camera_positions,
                              points_3d=None, path=PATH_TO_SAVE_COMPARISON_TO_GT_LOOP_CLOSURE + path_suffix, suffix="(LOOP CLOSURE)")
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_trajectories(camera_positions=self._final_estimated_trajectory, gt_camera_positions=gt_camera_positions,
                              points_3d=None, path=PATH_TO_SAVE_COMPARISON_TO_GT_LOOP_CLOSURE + path_suffix, suffix="(LOOP CLOSURE)")

    def get_absolute_localization_error(self, path_suffix=""):
        try:
            plot_localization_error_over_time(self.key_frames, camera_positions=self._final_estimated_trajectory,
                                              gt_camera_positions=self._gt_trajectory,
                                              path=PATH_TO_SAVE_LOCALIZATION_ERROR_LOOP_CLOSURE_AFTER + path_suffix,
                                              mode="Loop Closure")
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_localization_error_over_time(self.key_frames, camera_positions=self._final_estimated_trajectory,
                                              gt_camera_positions=self._gt_trajectory,
                                              path=PATH_TO_SAVE_LOCALIZATION_ERROR_LOOP_CLOSURE_AFTER + path_suffix,
                                              mode="Loop Closure")

    def show_uncertainty(self):
        try:
            plot_pg_uncertainty_before_and_after_lc(self._pose_graph, self._cur_pose_graph_estimates)
        except AttributeError as e:
            print(f"{e}.\nRunning solve_trajectory...")
            self.solve_trajectory()
            plot_pg_uncertainty_before_and_after_lc(self._pose_graph, self._cur_pose_graph_estimates)

    def get_final_estimated_trajectory(self):
        return get_trajectory_from_graph(self._cur_pose_graph_estimates)

    def get_successful_lc(self):
        return self._successful_lc

    def show_loop_between_two_keyframes(self, kf_1, kf_2, suffix=""):
        assert (kf_1, kf_2) in self._successful_lc or (kf_2, kf_1) in self._successful_lc, "Can only show existing loop..."
        from utils.plotters import plot_loop_between_two_frames
        plot_loop_between_two_frames(self._final_estimated_trajectory, kf_1, kf_2, self.key_frames, path=f"plots/lc_{suffix}_")

    def show_all_loops_on_trajectory(self, trajectory=None, suffix=""):
        from utils.plotters import plot_trajectory_with_loops
        if trajectory is None:
            trajectory = self._final_estimated_trajectory if self._final_estimated_trajectory is not None else self._initial_trajectory
        plot_trajectory_with_loops(trajectory, self._successful_lc, path=f"plots/lc_{suffix}_")

    def show_given_loops_on_trajectory(self, given_loops, trajectory=None, suffix="given_loops"):
        from utils.plotters import plot_trajectory_with_loops
        if trajectory is None:
            trajectory = self._final_estimated_trajectory if self._final_estimated_trajectory is not None else self._initial_trajectory
        plot_trajectory_with_loops(trajectory, given_loops, path=f"plots/lc_{suffix}_")

    def serialize(self, path=PATH_TO_SAVE_LOOP_CLOSURE_RESULTS):
        import pickle
        loop_closure_results_dict = {"initial_trajectory": self._initial_trajectory,
                                     "key_frames": self.key_frames,
                                     "successful_lc": self._successful_lc,
                                     "final_trajectory": self._final_estimated_trajectory,
                                     "good_mahalanobis_candidates": self._good_mahalanobis_candidates,
                                     "cur_pose_graph_estimates": self._cur_pose_graph_estimates}
        with open(path, 'wb') as f:
            pickle.dump(loop_closure_results_dict, f)

    @staticmethod
    def deserialize(lc_solver, path_to_deserialize):
        import pickle
        try:
            with open(path_to_deserialize, 'rb') as f:
                loop_closure_results_dict = pickle.load(f)
                lc_solver._initial_trajectory = loop_closure_results_dict["initial_trajectory"]
                lc_solver.key_frames = loop_closure_results_dict["key_frames"]
                lc_solver._successful_lc = loop_closure_results_dict["successful_lc"]
                lc_solver._final_estimated_trajectory = loop_closure_results_dict["final_trajectory"]
                lc_solver._good_mahalanobis_candidates = loop_closure_results_dict["good_mahalanobis_candidates"]
                lc_solver._cur_pose_graph_estimates = loop_closure_results_dict["cur_pose_graph_estimates"]
                lc_solver._gt_trajectory = get_gt_trajectory()[lc_solver.key_frames]
                print(f"Found File At {path_to_deserialize}, loaded loop closure results...")
            return SUCCESS
        except Exception as e:
            print('\033[91m' + f"Caught Exception When loading lc results: {e}" + '\033[0m')
            return FAILURE