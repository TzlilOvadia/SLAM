from models.TrackDatabase import TrackDatabase
from models.TrajectorySolver import PNP, BundleAdjustment, LoopClosure


if __name__ == "__main__":
    track_db = TrackDatabase()

    pnp_solver = PNP(track_db)
    pnp_solver.solve_trajectory()

    bundle_adjustment_solver = BundleAdjustment(track_db)
    bundle_adjustment_solver.solve_trajectory()

    loop_closure_solver = LoopClosure(track_db)
    loop_closure_solver.solve_trajectory()

    exit(0)
