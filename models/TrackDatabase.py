import numpy as np
from utils.utils import read_images
import pickle
import random
import models.Constants as Constants
# noinspection PyCompatibility
FEATURES = 1
EVEN = 0
ODD = 1

class TrackDatabase:
    def __init__(self, path_to_pkl_file=None):
        if path_to_pkl_file:
            self.deserialize(path_to_pkl_file)
            return
        self._tracks = {}  # Dictionary to store tracks
        self._frame_ids = {}  # Dictionary to store frame ids
        self.__next_free_trackId = 0
        self._last_insertions = {ODD: {}, EVEN: {}}  # Dictionary to store last left_kps indices (Keys) and a matching trackId (Values)
        self._num_tracks = 0
        self._num_frames = 0
        self._mean_track_length = 0
        self._max_length = 0
        self._min_length = np.inf
        self._mean_frame_links = 0
        self._frame_id_to_inliers_ratio = {}

    def add_track(self, trackId, frameId, feature_location_prev, feature_location_cur, kp_prev, kp_cur):
        prev_feature = (kp_prev, feature_location_prev, frameId)
        cur_feature = (kp_cur, feature_location_cur, frameId + 1)

        try:

            # We add a later feature among the two features, since we inserted both the previous and the current
            # features on the track setup stage. Thus, we shall add the current feature for sake of consistency.
            self._tracks[trackId].append(cur_feature)

        except KeyError:
            # On the setup of a new track, we add the
            self._tracks[trackId] = [prev_feature, cur_feature]
            self._frame_ids[frameId + 1].add(trackId)
            self._num_tracks += 1

        finally:
            self._frame_ids[frameId].add(trackId)
            self._last_insertions[(frameId + 1) % 2][kp_cur] = trackId
            self._num_frames = len(self._frame_ids.keys())
            self._max_length = max(self._max_length, len(self._tracks[trackId]))
            self._min_length = min(self._min_length, len(self._tracks[trackId]))

    def add_inliers_ratio(self, frame_id, ratio):
        assert frame_id in self._frame_ids
        self._frame_id_to_inliers_ratio[frame_id] = ratio

    def get_inliers_ratio_per_frame(self):
        return self._frame_id_to_inliers_ratio

    def get_track_ids_for_frame(self, frameId):
        """
        returns all the TrackIds that appear on a given FrameId.
        :param frameId:
        :return:
        """
        if frameId in self._frame_ids:
            return list(self._frame_ids[frameId])
        else:
            return []

    def get_frame_ids_for_track(self, trackId):
        if trackId in self._tracks:
            track = self._tracks[trackId]
            frame_ids_in_track = [track_point[2] for track_point in track]
            return frame_ids_in_track
        else:
            return []

    def get_feature_locations(self, frameId, trackId):
        if trackId in self._tracks:
            track = self._tracks[trackId]
            for track_point in track:
                if track_point[-1] == frameId:
                    return track_point[1]
        else:
            return None
    #
    # def extend_database(self, frameId, matches):
    #     for track_id, feature_location in matches.items():
    #         self.add_track(track_id, frameId, feature_location)

    def serialize(self, file_path):
        data = {
            'tracks': self._tracks,
            'frame_ids': self._frame_ids,
            'next_free_trackId': self.__next_free_trackId,
            'last_insertions': self._last_insertions,
            'num_tracks': self._num_tracks,
            'num_frames': self._num_frames,
            'mean_track_length': self._mean_track_length,
            'max_length': self._max_length,
            'min_length': self._min_length,
            'mean_frame_links': self._mean_frame_links,
            'frame_id_to_inliers_ratio': self._frame_id_to_inliers_ratio
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def deserialize(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self._tracks = data['tracks']
                self._frame_ids = data['frame_ids']
                self.__next_free_trackId = data['next_free_trackId']
                self._last_insertions = data['last_insertions']
                self._num_tracks = data['num_tracks']
                self._num_frames = data['num_frames']
                self._mean_track_length = data['mean_track_length']
                self._max_length = data['max_length']
                self._min_length = data['min_length']
                self._mean_frame_links = data['mean_frame_links']
                self._frame_id_to_inliers_ratio = data['frame_id_to_inliers_ratio']
            return Constants.SUCCESS
        except Exception:
            return Constants.FAILURE

    def generate_new_track_id(self):
        new_trackId = self.__next_free_trackId
        self.__next_free_trackId += 1
        return new_trackId

    def get_kp_trackId(self, kp, frameId):
        try:
            return self._last_insertions[frameId % 2][kp]
        except KeyError:
            return

    def prepare_to_next_pair(self, frameId):
        self._frame_ids[frameId] = set()
        self._frame_ids[frameId + 1] = set()
        self._last_insertions[(frameId + 1) % 2] = {}

    def get_num_tracks(self):
        return self._num_tracks

    def get_num_frames(self):
        return self._num_frames

    def get_mean_track_length(self):
        result = 0
        tracks_to_iterate_on = self._tracks.keys()
        for key in tracks_to_iterate_on:
            result += len(self._tracks[key])
        total_num_of_tracks = self._num_tracks
        try:
            self._mean_track_length = result / total_num_of_tracks
            return self._mean_track_length
        except ZeroDivisionError:
            return 0

    def get_max_track(self):
        return self._max_length

    def get_min_track(self):
        return self._min_length

    def get_mean_frame_links(self):
        result = 0
        for key in self._frame_ids.keys():
            result += len(self._frame_ids[key])

        try:
            self._mean_frame_links = result / self._num_frames
            return self._mean_frame_links
        except ZeroDivisionError:
            return 0

    def get_random_track_of_length(self, length):
        tracks_of_given_length = []
        for trackId, track in self._tracks.items():
            if len(track) >= length:
                tracks_of_given_length.append(track)
        if len(tracks_of_given_length) > 0:
            return random.choice(tracks_of_given_length)
        return None

    def calculate_connectivity_data(self):
        number_of_frames = self._num_frames
        x = np.arange(number_of_frames)
        y = np.zeros_like(x)
        for i in range(len(y) - 1):
            track_ids_in_frame = self._frame_ids[i]
            track_ids_in_next_frame = self._frame_ids[i+1]
            num_of_outgoing_tracks = 0
            for track_id in track_ids_in_frame:
                if track_id in track_ids_in_next_frame:
                    num_of_outgoing_tracks += 1
            y[i] = num_of_outgoing_tracks
        return x, y

    def get_track_length_data(self):
        track_lengths = [len(self._tracks[trackId]) for trackId in self._tracks.keys()]
        return np.array(track_lengths)
