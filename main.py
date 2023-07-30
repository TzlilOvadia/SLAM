from models.TrackDatabase import TrackDatabase
from models.TrajectorySolver import PNP, BundleAdjustment, LoopClosure


if __name__ == "__main__":
    track_db = TrackDatabase()

    print("#################### Starting PNP segment ####################")
    pnp_solver = PNP(track_db)
    pnp_solver.solve_trajectory()

    print("#################### Starting Bundle Adjustment segment ####################")
    bundle_adjustment_solver = BundleAdjustment(track_db)
    bundle_adjustment_solver.solve_trajectory()

    print("#################### Starting Loop Closure segment ####################")
    loop_closure_solver = LoopClosure(track_db)
    loop_closure_solver.solve_trajectory()

    exit(0)
