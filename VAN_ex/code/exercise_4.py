import numpy as np

from models.TrackDatabase import TrackDatabase
from utils import utils
from utils.plotters import plot_dict, plot_connectivity_graph, gen_hist, plot_reprojection_errors
from utils.utils import track_camera_for_many_images, project_point_on_image, visualize_track, least_squares
from models.Constants import *

# ------------------------------------------ANSWERS TO EXERCISE QUESTIONS ---------------------------------------

def q1(path):
    print("------------------------------------Q1-------------------------------------")
    print(f"LOOK AROUND, THE REQUIRED FUNCTIONS FOR Q1 ARE ALL IMPLEMENTED!")


def q2(path):
    print("------------------------------------Q2-------------------------------------")
    print("Printing some statistics of the tracks data in the database...")
    path_to_track_db_file = path
    track_db = TrackDatabase(path_to_pkl_file=path_to_track_db_file)
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
    print(f"Mean Number of Frame Links: {track_db.get_mean_frame_links()}")


def q3(path, num_to_show=10):
    print("------------------------------------Q3-------------------------------------")
    path_to_track_db_file = path
    track_db = TrackDatabase(path_to_pkl_file=path_to_track_db_file)
    length = num_to_show
    long_track = track_db.get_random_track_of_length(length=length)
    if long_track is None:
        print(f"No track of length at least {length} in data base!")
        return
    print(f"Sampled a track of length {len(long_track)}! \n Preparing to visualize tracking for {length} images...")
    visualize_track(long_track, num_to_show=num_to_show)


def q4(path):
    print("------------------------------------Q4-------------------------------------")
    path_to_track_db_file = path
    track_db = TrackDatabase(path_to_pkl_file=path_to_track_db_file)
    print("Calculating Connectivity Graph and Plotting it...")
    frame_nums, outgoint_tracks_in_frame = track_db.calculate_connectivity_data()
    plot_connectivity_graph(frame_nums, outgoint_tracks_in_frame)


def q5(path):
    print("------------------------------------Q5-------------------------------------")
    path_to_track_db_file = path
    track_db = TrackDatabase(path_to_pkl_file=path_to_track_db_file)
    print(f"Getting the Frame to Inliers Ratio Data and Plotting it...")
    inliers_ratio_dict = track_db.get_inliers_ratio_per_frame()
    plot_dict(inliers_ratio_dict, x_title='Frame Index', y_title='Inliers Ratio', title='Inliers Ratio Per Frame Index')



def q6(path):
    print("------------------------------------Q6-------------------------------------")
    path_to_track_db_file = path
    track_db = TrackDatabase(path_to_pkl_file=path_to_track_db_file)
    print(f"Getting the Track Length Data and Plotting it...")
    track_lengths = track_db.get_track_length_data()
    gen_hist(track_lengths, bins=len(np.unique(track_lengths)), title="Track Length Histogram", x="Track Length", y="Track #")


def q7(path, length=10):
    print("------------------------------------Q7-------------------------------------")
    path_to_track_db_file = path
    k, m1, m2 = utils.read_cameras()
    track_db = TrackDatabase(path_to_pkl_file=path_to_track_db_file)
    print(f"Looking for a Random Track of length at least {length}...")
    long_track = track_db.get_random_track_of_length(length=length)
    print(f"Found a Track of length {len(long_track)}...")
    print(f"Triangulating Feature Point from the Last/First Frame of the Path According to GT Camera Matrices...")
    gt_extrinsic_matrices = utils.read_gt()
    for frame, ind in {'last': -1, 'first': 0}.items():
        track_point_feature_location, track_point_frameId = long_track[ind][1], long_track[ind][2]
        p1 = track_point_feature_location[0], track_point_feature_location[2]
        p2 = track_point_feature_location[1], track_point_feature_location[2]
        Pmat = gt_extrinsic_matrices[track_point_frameId]
        Qmat = Pmat.copy()
        Qmat[:, -1] = Qmat[:, -1] + m2[:, -1]
        triangulated_3d_point = least_squares(p1, p2, k @ Pmat, k @ Qmat)
        print(f"Triangulated Point is:\n {triangulated_3d_point}")

        frame_ids = []
        left_errors = []
        right_errors = []
        print(f"Calculating the ReProjection Error for each Frames on The Track...")
        for i, track_point in enumerate(long_track):

            # Feature locations on images
            _, feature_location, frameId = track_point
            x_l, x_r, y = feature_location
            tracked_feature_location_left = np.array([x_l, y])
            tracked_feature_location_right = np.array([x_r, y])

            # Projected locations on images
            current_left_extrinsic_matrix = gt_extrinsic_matrices[frameId]
            current_right_extrinsic_matrix = current_left_extrinsic_matrix.copy()
            current_right_extrinsic_matrix[:, -1] = current_right_extrinsic_matrix[:, -1] + m2[:, -1]
            projected_point_on_left = project_point_on_image(triangulated_3d_point, current_left_extrinsic_matrix, k)
            projected_point_on_right = project_point_on_image(triangulated_3d_point, current_right_extrinsic_matrix, k)

            # calculating reprojection errors
            reprojection_error_left = np.linalg.norm(tracked_feature_location_left - projected_point_on_left)
            reprojection_error_right = np.linalg.norm(tracked_feature_location_right - projected_point_on_right)
            frame_ids.append(frameId)
            left_errors.append(reprojection_error_left)
            right_errors.append(reprojection_error_right)

        print(f"Plotting the ReProjection Errors...")
        plot_reprojection_errors(frame_ids, left_errors, right_errors, frame)


if __name__ == "__main__":
    PATH_TO_SAVE_TRACKER_FILE = "../../models/serialized_tracker_3"
    track_db = TrackDatabase()
    deserialization_result = track_db.deserialize(PATH_TO_SAVE_TRACKER_FILE)
    if deserialization_result == FAILURE:
        _, track_db = track_camera_for_many_images()
        track_db.serialize(PATH_TO_SAVE_TRACKER_FILE)
    print("finished serializing")
    q1(PATH_TO_SAVE_TRACKER_FILE)
    q2(PATH_TO_SAVE_TRACKER_FILE)
    q3(PATH_TO_SAVE_TRACKER_FILE, num_to_show=10)
    q4(PATH_TO_SAVE_TRACKER_FILE)
    q5(PATH_TO_SAVE_TRACKER_FILE)
    q6(PATH_TO_SAVE_TRACKER_FILE)
    q7(PATH_TO_SAVE_TRACKER_FILE, length=10)
