from models.Constants import *
from models.TrackDatabase import TrackDatabase
from utils.utils import track_camera_for_many_images, get_gt_trajectory
from models.TrajectorySolver import TrajectorySolver


class Slam:

    def __init__(self, solver: TrajectorySolver):
        self.__solver = solver
        self.__gt_trajectory = get_gt_trajectory()



    def solve_trajectory(self):
        self.__solver.solve_trajectory()

    def get_stats(self):
        self.__solver.get_stats()
