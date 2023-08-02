import random
from models.Constants import SEED
from models.TrackDatabase import TrackDatabase
from models.TrajectorySolver import PNP, BundleAdjustment, LoopClosure
from utils.plotters import plot_multiple_trajectories, plot_multiple_localization_error_over_time
from utils.utils import get_gt_trajectory

if __name__ == "__main__":
    random.seed(SEED)
    track_db = TrackDatabase()

    print("#################### Starting PNP segment ####################")
    pnp_solver = PNP(track_db)
    pnp_solver.solve_trajectory()
    # pnp_solver.get_absolute_localization_error()
    pnp_cam_pos = pnp_solver.get_final_estimated_trajectory()
    # pnp_solver.compare_trajectory_to_gt()
    print("#################### Starting Bundle Adjustment segment ####################")
    bundle_adjustment_solver = BundleAdjustment(track_db)
    bundle_adjustment_solver.solve_trajectory()
    bundle_adjustment_cam_pos = bundle_adjustment_solver.get_final_estimated_trajectory()
    bundle_adjustment_solver.get_absolute_localization_error()
    bundle_adjustment_solver.compare_trajectory_to_gt()
    bundle_adjustment_solver.get_rotation_error()
    print("#################### Starting Loop Closure segment ####################")
    loop_closure_solver = LoopClosure(track_db)
    loop_closure_solver.solve_trajectory()
    loop_closure_cam_pos = loop_closure_solver.get_final_estimated_trajectory()
    loop_closure_solver.compare_trajectory_to_gt()
    loop_closure_solver.get_absolute_localization_error()
    plot_multiple_trajectories(pnp_cam_pos, bundle_adjustment_cam_pos, loop_closure_cam_pos, get_gt_trajectory(),
                               path="plots/all_trajectories_comparison")
    exit(0)