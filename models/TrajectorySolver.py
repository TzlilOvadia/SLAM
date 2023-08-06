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

    def show_basic_tracking_statistics(self):
        track_db = self._track_db
        print("Printing some statistics of the tracks data in the database...")
        # Display Total Number of Tracks
        print(f"Total Number of Non-trivial Tracks: {track_db.get_num_tracks()}")
        # Display Total Number of Frames
        print(f"Total Number of Frames: {track_db.get_num_frames()}")
        # Display Mean Track Length
        print(f"Mean Track Length: {track_db.get_mean_track_length()}")
        # Display Maximal Track Length
        print(f"Maximal Track Length: {track_db.get_max_track()}")
        # Display Mininal Track Length
        print(f"Minimal Track Length: {track_db.get_min_track()}")
        # Display Mean Number of Frame Links
        print(f"Mean Number of Frame Links (number of tracks on an average image): {track_db.get_mean_frame_links()}")

    def show_connectivity_graph(self, path="connectivity_graph", suffix=""):
        from utils.plotters import plot_connectivity_graph
        track_db = self._track_db
        print("Calculating Connectivity Graph and Plotting it...")
        frame_nums, outgoint_tracks_in_frame = track_db.calculate_connectivity_data()
        plot_connectivity_graph(frame_nums, outgoint_tracks_in_frame, path=path + suffix)

    def show_inliers_ratio_graph(self, path="inliers_ratio_graph", suffix=""):
        from utils.plotters import plot_dict
        track_db = self._track_db
        print(f"Getting the Frame to Inliers Ratio Data and Plotting it...")
        inliers_ratio_dict = track_db.get_inliers_ratio_per_frame()
        plot_dict(inliers_ratio_dict, x_title='Frame Index', y_title='Inliers Ratio',
                  title='Inliers Ratio Per Frame Index', path=path + suffix)

    def show_num_matches_graph(self, path="num_matches_graph", suffix=""):
        from utils.plotters import plot_dict
        track_db = self._track_db
        print(f"Getting the Frame to Num Matches Data and Plotting it...")
        num_matches_dict = track_db.get_num_matches_per_frame()
        plot_dict(num_matches_dict, x_title='Frame Index', y_title='Num Matches',
                  title='Num Matches Per Frame Index', path=path + suffix)

    def show_track_length_histogram(self, path="track_length_histogram", suffix=""):
        from utils.plotters import gen_hist
        track_db = self._track_db
        print(f"Getting the Track Length Data and Plotting it...")
        track_lengths = track_db.get_track_length_data()
        gen_hist(track_lengths, bins=len(np.unique(track_lengths)), title="Track Length Histogram", x="Track Length",
                 y="Track #", path=path+suffix)

    def show_reprojection_error_for_tracks_at_given_length(self, length, path="reprojection_error_for_length_histogram", suffix=""):
        from utils.plotters import gen_hist
        from utils.utils import read_cameras, read_gt, least_squares, project_point_on_image
        track_db = self._track_db
        tracks = track_db.get_all_tracks_of_length(length=length)
        k, m1, m2 = read_cameras()
        print(f"Calculating reprojection error from last frame to first for tracks in length {length}...")
        gt_extrinsic_matrices = read_gt()
        reprojection_errors = []
        for trackId, track_data in tracks:
            # triangulating from last frame on track
            track_point_feature_location, track_point_frameId = track_data[-1][1], track_data[-1][2]
            p1 = track_point_feature_location[0], track_point_feature_location[2]
            p2 = track_point_feature_location[1], track_point_feature_location[2]
            Pmat = gt_extrinsic_matrices[track_point_frameId]
            Qmat = Pmat.copy()
            Qmat[:, -1] = Qmat[:, -1] + m2[:, -1]
            triangulated_3d_point = least_squares(p1, p2, k @ Pmat, k @ Qmat)

            # reprojecting from last frame to first frame
            _, feature_location, frameId = track_data[0]
            x_l, x_r, y = feature_location
            tracked_feature_location_left = np.array([x_l, y])
            current_left_extrinsic_matrix = gt_extrinsic_matrices[frameId]
            projected_point_on_left = project_point_on_image(triangulated_3d_point, current_left_extrinsic_matrix,
                                                             k)
            # calculating reprojection error
            reprojection_error_left = np.linalg.norm(tracked_feature_location_left - projected_point_on_left)
            reprojection_errors.append((trackId, reprojection_error_left))

        errors = np.array([re[1] for re in reprojection_errors])
        suffix = str(length) + "_" + suffix
        bins = len(np.unique(errors.astype(int)))
        gen_hist(errors, bins=bins, title=f"Reprojection Errors Histogram for Tracks of length {length}", x="Reprojection Error",
                 y="Track #", path=path+suffix)
        return reprojection_errors


    def show_reprojection_error_per_distance(self, length, path="pnp_reprojection_error_per_distance", suffix="PNP"):
        from utils.plotters import plot_median_projection_error_by_distance
        from utils.utils import read_cameras, read_gt, least_squares, project_point_on_image
        track_db = self._track_db
        tracks = track_db.get_all_tracks_of_length(length=length)
        k, m1, m2 = read_cameras()
        lengths_to_errors = [[] for i in range(length)]
        print(f"Calculating reprojection error per length {length} for PNP results...")
        extrinsic_matrices = track_db.get_extrinsic_matrices()
        for trackId, track_data in tracks:
            # triangulating from last frame on track
            track_point_feature_location, track_point_frameId = track_data[-1][1], track_data[-1][2]
            p1 = track_point_feature_location[0], track_point_feature_location[2]
            p2 = track_point_feature_location[1], track_point_feature_location[2]
            Pmat = extrinsic_matrices[track_point_frameId]
            Qmat = Pmat.copy()
            Qmat[:, -1] = Qmat[:, -1] + m2[:, -1]
            triangulated_3d_point = least_squares(p1, p2, k @ Pmat, k @ Qmat)

            # reprojecting from last frame to first frames
            for i, track_point in enumerate(track_data):
                _, feature_location, frameId = track_data[i]
                x_l, x_r, y = feature_location
                tracked_feature_location_left = np.array([x_l, y])
                current_left_extrinsic_matrix = extrinsic_matrices[frameId]
                projected_point_on_left = project_point_on_image(triangulated_3d_point, current_left_extrinsic_matrix,
                                                                 k)
                # calculating reprojection error
                reprojection_error_left = np.linalg.norm(tracked_feature_location_left - projected_point_on_left)
                distance = len(track_data) - i - 1
                lengths_to_errors[distance].append(reprojection_error_left)

        lengths_to_mean_errors = np.array([np.median(np.array(lengths_to_errors[i])) for i in range(length)])
        suffix = str(length) + "_" + suffix
        plot_median_projection_error_by_distance(lengths_to_mean_errors, path=path+suffix, title_suffix=suffix)
        return lengths_to_mean_errors



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

    def get_mean_factor_error_graph(self, path="mean_factor_error_graph", suffix=""):
        from utils.plotters import plot_mean_factor_error

        # calculate data

        initial_mean_factor_errors = []
        optimized_mean_factor_errors = []
        bundle_results = self.bundle_results[0]
        for single_bundle_results in bundle_results:
            i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates = single_bundle_results
            initial_total_graph_error = bundle_graph.error(initial_estimates)
            optimized_total_graph_error = bundle_graph.error(optimized_estimates)
            num_factors = bundle_graph.nrFactors()
            initial_mean_factor_errors.append(initial_total_graph_error / num_factors)
            optimized_mean_factor_errors.append(optimized_total_graph_error / num_factors)

        # plot results

        initial_mean_factor_errors = np.array(initial_mean_factor_errors)
        optimized_mean_factor_errors = np.array(optimized_mean_factor_errors)
        key_frames = self.key_frames[:-1]
        plot_mean_factor_error(initial_mean_factor_errors, optimized_mean_factor_errors, key_frames, path=path+suffix)

    def get_median_factor_error_graph(self, path="median_factor_error_graph", suffix=""):
        from utils.plotters import plot_median_factor_error

        # calculate data

        initial_median_factor_errors = []
        optimized_median_factor_errors = []
        bundle_results = self.bundle_results[0]
        for single_bundle_results in bundle_results:
            i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates = single_bundle_results
            initial_factor_errors = np.array([bundle_graph.at(i).error(initial_estimates) for i in range(bundle_graph.nrFactors())])
            optimized_factor_errors = np.array([bundle_graph.at(i).error(optimized_estimates) for i in range(bundle_graph.nrFactors())])
            initial_median = np.median(initial_factor_errors)
            optimized_median = np.median(optimized_factor_errors)
            initial_median_factor_errors.append(initial_median)
            optimized_median_factor_errors.append(optimized_median)

        # plot results

        initial_median_factor_errors = np.array(initial_median_factor_errors)
        optimized_median_factor_errors = np.array(optimized_median_factor_errors)
        key_frames = self.key_frames[:-1]
        plot_median_factor_error(initial_median_factor_errors, optimized_median_factor_errors, key_frames, path=path+suffix)


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