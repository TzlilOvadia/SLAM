from models.Slam import Slam
from models.TrackDatabase import TrackDatabase
from models.TrajectorySolver import PNP, BundleAdjustment, LoopClosure


if __name__ == "__main__":
    track_db = TrackDatabase()

    pnp_solver = PNP(track_db)
    pnp_slam = Slam(pnp_solver)
    pnp_slam.solve_trajectory()

    bundle_adjustment_solver = BundleAdjustment(track_db)
    bundle_slam = Slam(bundle_adjustment_solver)
    bundle_slam.solve_trajectory()


    loop_closure_solver = LoopClosure(track_db)
    loop_slam = Slam(loop_closure_solver)
    loop_slam.solve_trajectory()

    exit(0)