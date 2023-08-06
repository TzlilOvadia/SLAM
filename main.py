import random
from models.Constants import SEED
from models.TrackDatabase import TrackDatabase
from models.TrajectorySolver import PNP, BundleAdjustment, LoopClosure
from utils.plotters import plot_multiple_trajectories, plot_multiple_localization_error_over_time
from utils.utils import get_gt_trajectory

if __name__ == "__main__":
    random.seed(SEED)
    track_db = TrackDatabase()
    import sys

    print("Python version:", sys.version)
    print("#################### Starting PNP segment ####################")
    pnp_solver = PNP(force_recompute=False)
    #pnp_solver.show_reprojection_error_for_tracks_at_given_length(length=10)
    pnp_solver.solve_trajectory()
    # pnp_solver.get_absolute_localization_error()
    pnp_cam_pos = pnp_solver.get_final_estimated_trajectory()
    # pnp_solver.compare_trajectory_to_gt()
    print("#################### Starting Bundle Adjustment segment ####################")
    bundle_adjustment_solver = BundleAdjustment(force_recompute=False)
    bundle_adjustment_solver.solve_trajectory()
    bundle_adjustment_solver.get_median_factor_error_graph()
    bundle_adjustment_cam_pos = bundle_adjustment_solver.get_final_estimated_trajectory()
    #bundle_adjustment_solver.get_absolute_localization_error()
    #bundle_adjustment_solver.compare_trajectory_to_gt()
    #bundle_adjustment_solver.get_rotation_error()
    print("#################### Starting Loop Closure segment ####################")
    loop_closure_solver = LoopClosure()
    loop_closure_solver.solve_trajectory()
    #loop_closure_cam_pos = loop_closure_solver.get_final_estimated_trajectory()
    #loop_closure_solver.compare_trajectory_to_gt()
    #loop_closure_solver.get_absolute_localization_error()
    # plot_multiple_trajectories(pnp_cam_pos, bundle_adjustment_cam_pos, loop_closure_cam_pos, get_gt_trajectory(),
    #                            path="plots/all_trajectories_comparison")
    key_frames = loop_closure_solver.key_frames
    plot_multiple_trajectories(camera_positions_PNP=pnp_solver.get_final_estimated_trajectory()[key_frames],
                               camera_positions_bundle_adjustment=bundle_adjustment_solver.get_final_estimated_trajectory(),
                               camera_positions_loop_closure=loop_closure_solver.get_final_estimated_trajectory(),
                               gt_camera_positions=get_gt_trajectory()[key_frames],
                               suffix="")
    exit(0)
