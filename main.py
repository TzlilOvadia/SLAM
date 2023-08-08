import random
from models.Constants import SEED
from models.TrackDatabase import TrackDatabase
from models.TrajectorySolver import PNP, BundleAdjustment, LoopClosure
from utils.plotters import plot_multiple_trajectories, plot_multiple_localization_error_over_time, plot_multiple_angle_diffs
from utils.utils import get_gt_trajectory

if __name__ == "__main__":
    random.seed(SEED)
    track_db = TrackDatabase()
    import sys
    print("Python version:", sys.version)
    running_ver = "ver1"
    print("#################### Starting PNP segment ####################")
    pnp_solver = PNP(force_recompute=False)
    pnp_solver.solve_trajectory()
    pnp_solver.compare_trajectory_to_gt(path_suffix=running_ver)
    pnp_solver.get_absolute_localization_error(path_suffix=running_ver)
    pnp_solver.get_absolute_estimation_error(path="pnp_absolute_estimation_error" + running_ver, mode="PNP", invert_estimated_poses=False)
    pnp_solver.get_relative_estimation_error(mode="PNP", path_suffix="pnp" + running_ver, invert_estimated_poses=False)
    pnp_solver.show_basic_tracking_statistics()
    pnp_solver.show_num_matches_graph(suffix=running_ver)
    pnp_solver.show_inliers_ratio_graph(suffix=running_ver)
    pnp_solver.show_connectivity_graph(suffix=running_ver)
    pnp_solver.show_track_length_histogram(suffix=running_ver)

    print("#################### Starting Bundle Adjustment segment ####################")
    bundle_adjustment_solver = BundleAdjustment(force_recompute=False)
    bundle_adjustment_solver.solve_trajectory()
    bundle_adjustment_solver.compare_trajectory_to_gt(path_suffix=running_ver)
    bundle_adjustment_solver.get_absolute_localization_error(path_suffix=running_ver)
    bundle_adjustment_solver.get_absolute_estimation_error(path="bundle_adjustment_absolute_estimation_error" + running_ver, mode="BUNDLE ADJUSTMENT", invert_estimated_poses=True)
    bundle_adjustment_solver.get_relative_estimation_error(mode="BUNDLE_ADJUSTMENT", path_suffix="bundle_adjustment" + running_ver,invert_estimated_poses=True)
    bundle_adjustment_solver.get_mean_factor_error_graph(suffix=running_ver)
    bundle_adjustment_solver.get_median_factor_error_graph(suffix=running_ver)
    bundle_adjustment_solver.show_factor_errors_and_reprojection_errors_by_distance(length=15, path_suffix=running_ver)

    print("#################### Starting Loop Closure segment ####################")
    loop_closure_solver = LoopClosure(force_recompute=False)
    loop_closure_solver.solve_trajectory()
    loop_closure_solver.show_all_loops_on_trajectory(suffix=running_ver)
    loop_closure_solver.compare_trajectory_to_gt(path_suffix=running_ver)
    loop_closure_solver.get_absolute_localization_error(path_suffix=running_ver)
    loop_closure_solver.get_absolute_estimation_error(path="loop_closure_absolute_estimation_error" + running_ver, mode="LOOP_CLOSURE", invert_estimated_poses=True)
    loop_closure_solver.get_relative_estimation_error(mode="LOOP_CLOSURE", path_suffix="loop_closure" + running_ver, invert_estimated_poses=True)
    loop_closure_solver.show_num_matches_and_inliers_per_lc(path_suffix=running_ver)
    loop_closure_solver.show_uncertainty(path_suffix=running_ver)

    key_frames = loop_closure_solver.key_frames
    plot_multiple_trajectories(camera_positions_PNP=pnp_solver.get_final_estimated_trajectory()[key_frames],
                               camera_positions_bundle_adjustment=bundle_adjustment_solver.get_final_estimated_trajectory(),
                               camera_positions_loop_closure=loop_closure_solver.get_final_estimated_trajectory(),
                               gt_camera_positions=get_gt_trajectory()[key_frames],
                               suffix=running_ver)
    a=5
