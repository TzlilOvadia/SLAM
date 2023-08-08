import numpy as np
from models.Constants import *
from models.Matcher import Matcher
from models.TrackDatabase import TrackDatabase
from models.BundleAdjustment import bundle_adjustment, create_pose_graph, load_bundle_results, get_translation_rotation_diff
from utils.utils import track_camera_for_many_images, get_gt_trajectory
from utils.plotters import plot_trajectories, plot_localization_error_over_time
from models.LoopClosure import loop_closure, plot_pg_locations_before_and_after_lc,\
    plot_pg_locations_error_graph_before_and_after_lc, plot_pg_uncertainty_before_and_after_lc, get_trajectory_from_graph, get_poses_from_graph


class TrajectorySolver:

    def __init__(self, path_to_track_db=PATH_TO_SAVE_TRACKER_FILE):
        self.__matcher = Matcher()
        self._final_estimated_trajectory = None
        self.deserialization_result = None
        self._track_db = TrackDatabase()
        self._load_tracks_to_db(path=path_to_track_db)
        self.key_frames = None
        self._gt_trajectory = self.get_gt_trajectory()
        self._predicted_trajectory = None



    def get_rotation_error(self):
        raise NotImplementedError("Subclasses should implement this!")

    def solve_trajectory(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_absolute_localization_error(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_angle_diff(self, invert_estimated_poses=False):
        from utils.utils import get_rotation_matrices_distances
        camera_gt_poses = self.get_gt_poses()
        camera_estimated_poses = self.get_estimated_poses_matrices(invert_estimated_poses=invert_estimated_poses)
        gt_rotations = camera_gt_poses[:, :, :-1]
        angles_diff = get_rotation_matrices_distances(camera_estimated_poses[:, :, :-1], gt_rotations)
        return angles_diff

    def get_angle_diff_between_given_rotations(self, rotation_1, rotation_2):
        from utils.utils import get_rotation_matrices_distances
        angles_diff = get_rotation_matrices_distances(rotation_1, rotation_2)
        return angles_diff

    def get_xyz_diffs(self):
        camera_estimated_xyz_positions = self.get_final_estimated_trajectory()
        camera_gt_xyz_positions = self.get_gt_trajectory()
        xyz_positions_diff = np.abs(camera_gt_xyz_positions - camera_estimated_xyz_positions)
        x_error, y_error, z_error = xyz_positions_diff[:, 0], xyz_positions_diff[:, 1], xyz_positions_diff[:, 2]
        total_norm_error = np.linalg.norm(xyz_positions_diff, axis=1)
        return x_error, y_error, z_error, total_norm_error

    def get_absolute_estimation_error(self, path, mode, invert_estimated_poses=False):
        from utils.plotters import plot_absolute_xyz_location_diff, plot_absolute_angle_diff_composition
        key_frames = self.key_frames
        x_error, y_error, z_error, total_norm_error = self.get_xyz_diffs()
        angles_diff = self.get_angle_diff(invert_estimated_poses)
        plot_absolute_xyz_location_diff(x_error, y_error, z_error, total_norm_error, key_frames, path=path + "_location_", mode=mode)
        plot_absolute_angle_diff_composition(angles_diff, key_frames, path=path + "_angles_", mode=mode)


    def get_relative_estimation_error(self, sequence_lengths=(100, 300, 500, 800), mode="", path_suffix="", invert_estimated_poses=False):
        from utils.plotters import plot_relative_location_estimation_error, plot_relative_angle_estimation_error
        from utils.utils import angle_between
        from models.BundleAdjustment import get_relative_transformation_same_source_cs
        estimated_global_poses = self.get_estimated_poses_matrices(invert_estimated_poses=invert_estimated_poses)
        gt_global_poses = self.get_gt_poses()
        estimated_camera_positions = self.get_final_estimated_trajectory()
        gt_camera_positions = self.get_gt_trajectory()
        gt_camera_positions_diffs = np.linalg.norm(gt_camera_positions[1:] - gt_camera_positions[:-1], axis=1)
        relative_dist_estimations = []
        relative_angle_estimations = []
        for length in sequence_lengths:
            kf_sequences = self.get_key_frames_sequences(length)
            estimated_relative_poses = np.array([get_relative_transformation_same_source_cs(estimated_global_poses[j],
                                                                                            estimated_global_poses[i])
                                                 for i, j in kf_sequences])

            gt_relative_poses = np.array([get_relative_transformation_same_source_cs(gt_global_poses[j],
                                                                                     gt_global_poses[i])
                                          for i, j in kf_sequences])


            estimated_relative_locations = np.array([estimated_camera_positions[i] - estimated_camera_positions[j]
                                                     for i, j in kf_sequences])

            gt_relative_locations = np.array([gt_camera_positions[i] - gt_camera_positions[j]
                                                     for i, j in kf_sequences])

            dist_denominator = np.array([np.sum(gt_camera_positions_diffs[i: j]) for i, j in kf_sequences])
            dist_numerator = np.linalg.norm(estimated_relative_locations - gt_relative_locations, axis=1)
            relative_dist_estimation = dist_numerator / dist_denominator
            key_frames = self.key_frames[:len(kf_sequences)]
            mean_relative_dist_estimation = np.mean(relative_dist_estimation)
            relative_dist_estimations.append((length, relative_dist_estimation, key_frames, mean_relative_dist_estimation))


            # angle_diffs = np.array([angle_between(estimated_relative_locations[i], gt_relative_locations[i])
            #                                       for i in range(len(estimated_relative_locations))])

            estimated_relative_rotations = estimated_relative_poses[:, :, :-1]
            gt_relative_rotations = gt_relative_poses[:, :, :-1]
            angle_diffs = self.get_angle_diff_between_given_rotations(estimated_relative_rotations, gt_relative_rotations)

            angle_diffs = np.rad2deg(angle_diffs)
            relative_angle_estimation = angle_diffs / dist_denominator
            mean_relative_angle_estimation = np.mean(relative_angle_estimation)
            relative_angle_estimations.append((length, relative_angle_estimation, key_frames, mean_relative_angle_estimation))
        plot_relative_location_estimation_error(relative_dist_estimations, mode=mode, path="relative_location_error_" + path_suffix)
        plot_relative_angle_estimation_error(relative_angle_estimations, mode=mode, path="relative_angle_error_" + path_suffix)


    def compare_trajectory_to_gt(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_gt_poses(self):
        from utils.utils import read_gt
        return read_gt()

    def get_estimated_poses_matrices(self, invert_estimated_poses=False):
        raise NotImplementedError("Subclasses should implement this!")

    def get_gt_trajectory(self):
        return get_gt_trajectory()

    def get_final_estimated_trajectory(self):
        """
        Getter method for camera positions.

        Returns:
            list: List of camera positions.
        """
        return self._final_estimated_trajectory

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

    def get_key_frames_sequences(self, length):
        import bisect
        key_frames = self.key_frames
        res = [(i, bisect.bisect_right(key_frames, kf + length-0.001)) for i, kf in enumerate(key_frames) if bisect.bisect_right(key_frames, kf + length-0.001) < len(key_frames)]
        return res

    def get_trajectory_from_poses(self, poses):
        trajectory = np.array([pose.translation() for pose in poses])
        return trajectory


class PNP(TrajectorySolver):



    def __init__(self, force_recompute=False, path_to_save_track_db=PATH_TO_SAVE_TRACKER_FILE):
        super().__init__(path_to_track_db=path_to_save_track_db)
        self.force_recompute = force_recompute
        self._final_estimated_trajectory = self._track_db.camera_positions
        self.key_frames = np.arange(self._track_db.get_num_frames())


    def compare_trajectory_to_gt(self, path_suffix=""):
        plot_trajectories(self._track_db.camera_positions, self._gt_trajectory, path=PATH_TO_SAVE_COMPARISON_TO_GT_PNP + path_suffix,
                          suffix="(PNP)")

    def solve_trajectory(self):
        if self.force_recompute or self.get_deserialization_result() != SUCCESS:
            _, self._track_db = track_camera_for_many_images()
            self._final_estimated_trajectory = self._track_db.camera_positions

    def get_estimated_poses_matrices(self, invert_estimated_poses=False):
        return self._track_db.get_extrinsic_matrices()


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

    def show_connectivity_graph(self, path="pnp_connectivity_graph", suffix=""):
        from utils.plotters import plot_connectivity_graph
        track_db = self._track_db
        print("Calculating Connectivity Graph and Plotting it...")
        frame_nums, outgoint_tracks_in_frame = track_db.calculate_connectivity_data()
        plot_connectivity_graph(frame_nums, outgoint_tracks_in_frame, path=path + suffix)

    def show_inliers_ratio_graph(self, path="pnp_inliers_ratio_graph", suffix=""):
        from utils.plotters import plot_dict
        track_db = self._track_db
        print(f"Getting the Frame to Inliers Ratio Data and Plotting it...")
        inliers_ratio_dict = track_db.get_inliers_ratio_per_frame()
        plot_dict(inliers_ratio_dict, x_title='Frame Index', y_title='Inliers Ratio',
                  title='Inliers Ratio Per Frame Index', path=path + suffix)

    def show_num_matches_graph(self, path="pnp_num_matches_graph", suffix=""):
        from utils.plotters import plot_dict
        track_db = self._track_db
        print(f"Getting the Frame to Num Matches Data and Plotting it...")
        num_matches_dict = track_db.get_num_matches_per_frame()
        plot_dict(num_matches_dict, x_title='Frame Index', y_title='Num Matches',
                  title='Num Matches Per Frame Index', path=path + suffix)

    def show_track_length_histogram(self, path="pnp_track_length_histogram", suffix=""):
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
        self.force_recompute = force_recompute
        self.path_to_save_db = path_to_save_track_db
        self.path_to_save_ba = path_to_save_ba
        self.optimized_estimates = None
        self._pose_graph = None

    def solve_trajectory(self):
        self.bundle_results = load_bundle_results(path=self.path_to_save_ba, force_recompute=self.force_recompute, track_db_path=self.path_to_save_db)
        bundle_windows = self.bundle_results[-2]
        self.key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
        bundle_results, optimized_relative_keyframes_poses, optimized_global_keyframes_poses, bundle_windows, \
        cond_matrices = self.bundle_results
        key_frames = [window[0] for window in bundle_windows] + [bundle_windows[-1][1]]
        self._pose_graph, self.optimized_estimates, _ = create_pose_graph(bundle_results,
                                                                    optimized_relative_keyframes_poses,
                                                                    optimized_global_keyframes_poses,
                                                                    cond_matrices)
        self._final_estimated_trajectory = get_trajectory_from_graph(self.optimized_estimates)
        self._gt_trajectory = get_gt_trajectory()[key_frames]
        self.optimized_global_keyframes_poses = optimized_global_keyframes_poses
        a=5

    def get_resulting_pose_graph_data(self):
        return self._pose_graph, self.optimized_estimates

    def get_gt_trajectory(self):
        if self.key_frames is None:
            return []
        return get_gt_trajectory()[self.key_frames]

    def get_gt_poses(self):
        from utils.utils import read_gt
        return read_gt()[self.key_frames]

    def get_estimated_poses_matrices(self, invert_estimated_poses=False):
        from utils.utils import invert_Rt_transformation
        poses = get_poses_from_graph(self.optimized_estimates)
        if invert_estimated_poses:
            poses = np.array([invert_Rt_transformation(Rt) for Rt in poses])
        return poses

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

    def show_factor_errors_and_reprojection_errors_by_distance(self, length, path_suffix=""):
        from utils.plotters import plot_all_median_projection_error_by_distance
        import gtsam
        import utils
        from models.BundleAdjustment import get_only_relevant_tracks_all
        K, M1, M2 = utils.utils.read_cameras()
        GTSAM_K = utils.utils.get_gtsam_calib_mat(K, M2)
        track_db = self.get_track_db()
        bundle_results = self.bundle_results[0]
        dists_to_reprojection_errors_initial = [[] for i in range(length)]
        dists_to_factor_errors_initial = [[] for i in range(length)]

        dists_to_reprojection_errors_optimized = [[] for i in range(length)]
        dists_to_factor_errors_optimized = [[] for i in range(length)]

        for single_bundle_results in bundle_results:
            i, bundle_window, bundle_graph, initial_estimates, landmarks, optimized_estimates = single_bundle_results
            bundle_starts_in_frame_id, bundle_ends_in_frame_id = bundle_window
            tracks = get_only_relevant_tracks_all(track_db, bundle_starts_in_frame_id, bundle_ends_in_frame_id)
            for track_data, trackId in tracks:
                point_symbol = gtsam.symbol(POINT, trackId)
                track_ends_in_frame_id = track_data[LAST_ITEM][FRAME_ID]
                track_starts_in_frame_id = track_data[0][FRAME_ID]
                track_length_in_bundle = min(track_ends_in_frame_id, bundle_ends_in_frame_id) - max(bundle_starts_in_frame_id, track_starts_in_frame_id) + 1
                if point_symbol not in landmarks or track_length_in_bundle != length:
                    continue
                offset = track_data[0][FRAME_ID]
                frameId_of_last_frame_of_track_in_bundle = min(bundle_ends_in_frame_id, track_ends_in_frame_id)
                frameId_of_first_frame_of_track_in_bundle = max(bundle_starts_in_frame_id, track_starts_in_frame_id)
                last_point3_initial = initial_estimates.atPoint3(point_symbol)
                last_point3_optimized = initial_estimates.atPoint3(point_symbol)

                for j, frame_id in enumerate(range(frameId_of_first_frame_of_track_in_bundle, frameId_of_last_frame_of_track_in_bundle + 1)):
                    cam_symbol = gtsam.symbol(CAMERA, frame_id)
                    index_of_relevant_track_point = frame_id - offset
                    location = track_data[index_of_relevant_track_point][LOCATIONS_IDX]
                    measured_point2 = gtsam.StereoPoint2(location[0], location[1], location[2])
                    stereomodel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
                    distance_from_reference_frame = frameId_of_last_frame_of_track_in_bundle - frame_id
                    factor = gtsam.GenericStereoFactor3D(measured_point2, stereomodel_noise, cam_symbol, point_symbol,
                                                         GTSAM_K)
                    factor_error_initial = factor.error(initial_estimates)
                    factor_error_optimized = factor.error(optimized_estimates)

                    current_cam_pose_initial = initial_estimates.atPose3(cam_symbol)
                    current_cam_pose_optimized = optimized_estimates.atPose3(cam_symbol)

                    projected_point_initial = gtsam.StereoCamera(current_cam_pose_initial, GTSAM_K).project(last_point3_initial)
                    projected_point_optimized = gtsam.StereoCamera(current_cam_pose_optimized, GTSAM_K).project(last_point3_optimized)

                    reprojection_error_initial = np.linalg.norm(np.array([measured_point2.uL(), measured_point2.v()]) - np.array([projected_point_initial.uL(), projected_point_initial.v()]))
                    reprojection_error_optimized = np.linalg.norm(np.array([measured_point2.uL(), measured_point2.v()]) - np.array([projected_point_optimized.uL(), projected_point_optimized.v()]))

                    dists_to_reprojection_errors_initial[distance_from_reference_frame].append(reprojection_error_initial)
                    dists_to_factor_errors_initial[distance_from_reference_frame].append(factor_error_initial)

                    dists_to_reprojection_errors_optimized[distance_from_reference_frame].append(reprojection_error_optimized)
                    dists_to_factor_errors_optimized[distance_from_reference_frame].append(factor_error_optimized)

        dists_to_median_reprojection_error_initial = np.array([np.median(np.array(dists_to_reprojection_errors_initial[i])) for i in range(length)])
        dists_to_median_factor_error_initial = np.array([np.median(np.array(dists_to_factor_errors_initial[i])) for i in range(length)])
        dists_to_median_reprojection_error_optimized = np.array([np.median(np.array(dists_to_reprojection_errors_optimized[i])) for i in range(length)])
        dists_to_median_factor_error_optimized = np.array([np.median(np.array(dists_to_factor_errors_optimized[i])) for i in range(length)])
        suffix1 = str(length) + "_median" + path_suffix
        plot_all_median_projection_error_by_distance(initial_rep_errors=dists_to_median_reprojection_error_initial,
                                                     optimized_rep_errors=dists_to_median_reprojection_error_optimized,
                                                     initial_fac_errors=dists_to_median_factor_error_initial,
                                                     optimized_fac_errors=dists_to_median_factor_error_optimized,
                                                     path_suffix=suffix1, statistic_type="Median ")

        dists_to_mean_reprojection_error_initial = np.array([np.mean(np.array(dists_to_reprojection_errors_initial[i])) for i in range(length)])
        dists_to_mean_factor_error_initial = np.array([np.mean(np.array(dists_to_factor_errors_initial[i])) for i in range(length)])
        dists_to_mean_reprojection_error_optimized = np.array([np.mean(np.array(dists_to_reprojection_errors_optimized[i])) for i in range(length)])
        dists_to_mean_factor_error_optimized = np.array([np.mean(np.array(dists_to_factor_errors_optimized[i])) for i in range(length)])
        suffix2 = str(length) + "_mean" + path_suffix
        plot_all_median_projection_error_by_distance(initial_rep_errors=dists_to_mean_reprojection_error_initial,
                                                     optimized_rep_errors=dists_to_mean_reprojection_error_optimized,
                                                     initial_fac_errors=dists_to_mean_factor_error_initial,
                                                     optimized_fac_errors=dists_to_mean_factor_error_optimized,
                                                     path_suffix=suffix2, statistic_type="Mean ")

        return

class LoopClosure(TrajectorySolver):

    def __init__(self,  path_to_save_track_db=PATH_TO_SAVE_TRACKER_FILE,
                 path_to_save_ba=PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS,
                 path_to_save_lc=PATH_TO_SAVE_LOOP_CLOSURE_RESULTS,
                 force_recompute=False):
        super().__init__(path_to_track_db=path_to_save_track_db)
        self._global_3d_points_numpy = None
        self._global_Rt_poses_in_numpy = None
        self._optimized_pose_graph = None
        self._initial_pose_graph = None
        self._initial_pose_graph_estimates = None
        self._initial = None
        self._optimized_pose_graph_estimates = None
        self._successful_lc = None
        self.bundle_results = None
        self.optimized_global_keyframes_poses = []
        self.bundle_3d_points = None
        self.optimized_relative_keyframes_poses = []
        self.global_3d_points = []
        self._good_mahalanobis_candidates = None
        self._initial_trajectory = None
        self._lc_to_matches_statistics = None
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
            self._initial_pose_graph = pose_graph.clone()
            self._initial_pose_graph_estimates = initial_estimates
            cond_matrices = [cond_matrix for cond_matrix in cond_matrices]
            self._initial_trajectory = self.get_trajectory_from_poses(optimized_global_keyframes_poses)
            self._optimized_pose_graph, self._optimized_pose_graph_estimates, self._successful_lc, good_ms, lc_to_matches_statistics\
                = loop_closure(pose_graph, self.key_frames,
                               matcher=super().get_matcher(), cond_matrices=cond_matrices,
                               mahalanobis_thresh=MAHALANOBIS_THRESH,
                               pose_graph_initial_estimates=initial_estimates,
                               draw_supporting_matches_flag=False,points_to_stop_by=False
                               )
            self._good_mahalanobis_candidates = [(c[1], c[2]) for c in good_ms]
            self._lc_to_matches_statistics = lc_to_matches_statistics
            self._final_estimated_trajectory = get_trajectory_from_graph(self._optimized_pose_graph_estimates)
            self._gt_trajectory = get_gt_trajectory()[self.key_frames]
            self.serialize(self.path_to_save_lc)

    def get_estimated_poses_matrices(self, invert_estimated_poses=False):
        from utils.utils import invert_Rt_transformation
        poses = get_poses_from_graph(self._optimized_pose_graph_estimates)
        if invert_estimated_poses:
            poses = np.array([invert_Rt_transformation(Rt) for Rt in poses])
        return poses

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

    def show_uncertainty(self, path_suffix=""):
        import gtsam
        from utils.plotters import plot_uncertainty_over_time, plot_combined_uncertainty_over_time
        key_frames = self.key_frames
        pose_graph_after = self._optimized_pose_graph
        estimates_after = self._optimized_pose_graph_estimates
        pose_graph_before = self._initial_pose_graph
        estimates_before = self._initial_pose_graph_estimates

        optimizer = gtsam.LevenbergMarquardtOptimizer(pose_graph_before, estimates_before)
        estimates_before = optimizer.optimize()

        pose_graph_after_covariance = gtsam.Marginals(pose_graph_after, estimates_after)
        pose_graph_before_covariance = gtsam.Marginals(pose_graph_before, estimates_before)
        total_after = []
        total_before = []
        location_after = []
        location_before = []
        angle_after = []
        angle_before = []
        for kf in key_frames:
            key_vectors = gtsam.KeyVector()
            key_vectors.append(gtsam.symbol(CAMERA, kf))
            total_kf_cov_after = pose_graph_after_covariance.jointMarginalCovariance(key_vectors).fullMatrix()
            location_kf_cov_after = total_kf_cov_after[3:, 3:]
            angle_kf_cov_after = total_kf_cov_after[:3, :3]

            total_kf_cov_before = pose_graph_before_covariance.jointMarginalCovariance(key_vectors).fullMatrix()
            location_kf_cov_before = total_kf_cov_before[3:, 3:]
            angle_kf_cov_before = total_kf_cov_before[:3, :3]

            total_after.append(total_kf_cov_after)
            total_before.append(total_kf_cov_before)
            location_after.append(location_kf_cov_after)
            location_before.append(location_kf_cov_before)
            angle_after.append(angle_kf_cov_after)
            angle_before.append(angle_kf_cov_before)

        total_after_score = [np.abs(np.linalg.det(cov)) for cov in total_after]
        total_before_score = [np.abs(np.linalg.det(cov)) for cov in total_before]
        location_after_score = [np.abs(np.linalg.det(cov)) for cov in location_after]
        location_before_score = [np.abs(np.linalg.det(cov)) for cov in location_before]
        angle_after_score = [np.abs(np.linalg.det(cov)) for cov in angle_after]
        angle_before_score = [np.abs(np.linalg.det(cov)) for cov in angle_before]

        frames_with_loops = self.get_frameIds_of_loop_closures()
        plot_uncertainty_over_time(key_frames, total_after_score, "plots/LOOP_CLOSURE_uncertainty_" + path_suffix, title_suffix="(LOOP CLOSURE)", loop_closures_frames=frames_with_loops, kind="Total")
        plot_uncertainty_over_time(key_frames, total_before_score, "plots/BUNDLE_ADJUSTMENT_uncertainty_" + path_suffix, title_suffix="(BUNDLE ADJUSTMENT)", kind="Total")

        plot_uncertainty_over_time(key_frames, location_before_score, "plots/BUNDLE_ADJUSTMENT_uncertainty_location" + path_suffix, title_suffix="(BUNDLE ADJUSTMENT)", kind="Location")
        plot_uncertainty_over_time(key_frames, angle_before_score, "plots/BUNDLE_ADJUSTMENT_uncertainty_angle" + path_suffix, title_suffix="(BUNDLE ADJUSTMENT)", kind="Angle")
        plot_uncertainty_over_time(key_frames, location_after_score, "plots/LOOP_CLOSURE_uncertainty_location" + path_suffix, title_suffix="(LOOP_CLOSURE)", loop_closures_frames=frames_with_loops, kind="Location")
        plot_uncertainty_over_time(key_frames, angle_after_score, "plots/LOOP_CLOSURE_uncertainty_angle" + path_suffix, title_suffix="(LOOP_CLOSURE)", loop_closures_frames=frames_with_loops, kind="Angle")


        plot_combined_uncertainty_over_time(key_frames, np.log(total_before_score)+np.abs(np.min(np.log(total_before_score))),
                                            np.log(total_after_score)+np.abs(np.min(np.log(total_before_score))),
                                            "plots/log_combined_uncertainty_" + path_suffix,
                                            title_suffix="(BA and LC- log)", loop_closures_frames=frames_with_loops)
        return

    def get_frameIds_of_loop_closures(self):
        frameIds = set()
        for a, b in self._successful_lc:
            frameIds.add(self.key_frames[a])
            frameIds.add(self.key_frames[b])
        frameIds = list(frameIds)
        frameIds.sort()
        return frameIds

    def get_gt_trajectory(self):
        if self.key_frames is None:
            return []
        return get_gt_trajectory()[self.key_frames]

    def get_gt_poses(self):
        from utils.utils import read_gt
        return read_gt()[self.key_frames]

    def get_lc_statistics(self):
        return self._lc_to_matches_statistics

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
        plot_trajectory_with_loops(trajectory, self._successful_lc, path_suffix=suffix)

    def show_given_loops_on_trajectory(self, given_loops, trajectory=None, suffix="given_loops"):
        from utils.plotters import plot_trajectory_with_loops
        if trajectory is None:
            trajectory = self._final_estimated_trajectory if self._final_estimated_trajectory is not None else self._initial_trajectory
        plot_trajectory_with_loops(trajectory, given_loops, path_suffix=suffix)

    def show_num_matches_and_inliers_per_lc(self, path_suffix=""):
        from utils.plotters import plot_lc_inlier_ratio_and_md
        lc_stats_dict = self.get_lc_statistics()
        lc_stats = list(lc_stats_dict.values())
        lc_stats.sort(key=lambda e: e[0])
        m_dists = np.array([lc_stat[0] for lc_stat in lc_stats])
        num_matches = np.array([lc_stat[1] for lc_stat in lc_stats])
        inlier_ratios = np.array([lc_stat[2] for lc_stat in lc_stats])
        plot_lc_inlier_ratio_and_md(m_dists, num_matches, inlier_ratios, path_suffix=path_suffix)
        return

    def serialize(self, path=PATH_TO_SAVE_LOOP_CLOSURE_RESULTS):
        import pickle
        loop_closure_results_dict = {"initial_trajectory": self._initial_trajectory,
                                     "key_frames": self.key_frames,
                                     "successful_lc": self._successful_lc,
                                     "final_trajectory": self._final_estimated_trajectory,
                                     "good_mahalanobis_candidates": self._good_mahalanobis_candidates,
                                     "cur_pose_graph_estimates": self._optimized_pose_graph_estimates,
                                     "lc_to_matches_statistics": self._lc_to_matches_statistics,
                                     "initial_pose_graph": self._initial_pose_graph,
                                     "optimized_pose_graph": self._optimized_pose_graph,
                                     "initial_estimates": self._initial_pose_graph_estimates}
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
                lc_solver._optimized_pose_graph_estimates = loop_closure_results_dict["cur_pose_graph_estimates"]
                lc_solver._gt_trajectory = get_gt_trajectory()[lc_solver.key_frames]
                if "lc_to_matches_statistics" in loop_closure_results_dict:
                    lc_solver._lc_to_matches_statistics = loop_closure_results_dict["lc_to_matches_statistics"]
                if "initial_pose_graph" in loop_closure_results_dict:
                    lc_solver._initial_pose_graph = loop_closure_results_dict["initial_pose_graph"]
                if "optimized_pose_graph" in loop_closure_results_dict:
                    lc_solver._optimized_pose_graph = loop_closure_results_dict["optimized_pose_graph"]
                if "initial_estimates" in loop_closure_results_dict:
                    lc_solver._initial_pose_graph_estimates = loop_closure_results_dict["initial_estimates"]
                print(f"Found File At {path_to_deserialize}, loaded loop closure results...")
            return SUCCESS
        except Exception as e:
            print('\033[91m' + f"Caught Exception When loading lc results: {e}" + '\033[0m')
            return FAILURE