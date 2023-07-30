from models.Constants import *
from models.TrackDatabase import TrackDatabase
from models.BundleAdjustment import bundle_adjustment, create_pose_graph, load_bundle_results
from utils.utils import track_camera_for_many_images, get_gt_trajectory, plot_trajectories
from utils.plotters import plot_trajectories
from LoopClosure import loop_closure
from TrajectorySolver import TrajectorySolver


class Slam:

    def __init__(self, solver: TrajectorySolver):
        self.__solver = solver
        self.__track_db = TrackDatabase()
        self.__gt_trajectory = get_gt_trajectory()

    def load_tracks_to_db(self):
        deserialization_result = self.track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
        if deserialization_result == FAILURE:
            _, self.track_db = track_camera_for_many_images()
            self.track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)

    def solve_trajectory(self):
        self.__solver.solve_trajectory()

    def get_stats(self):
        self.__solver.get_stats()
