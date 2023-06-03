# noinspection PyCompatibility
FEATURES = 1
EVEN = 0
ODD = 1

class TrackDatabase:
    def __init__(self):
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

    def add_track(self, trackId, frameId, feature_location_prev, kp_prev, kp_cur):
        prev_feature = (kp_prev, feature_location_prev, frameId)

        try:
            self._tracks[trackId].append(prev_feature)

        except KeyError:
            self._tracks[trackId] = [prev_feature]
            self._num_tracks += 1

        finally:
            self._frame_ids[frameId].add(trackId)
            self._last_insertions[(frameId + 1) % 2][kp_cur] = trackId
            self._num_frames = len(self._frame_ids.keys())
            self._max_length = max(self._max_length, len(self._tracks[trackId]))
            self._min_length = min(self._min_length, len(self._tracks[trackId]))


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
        if track_id in self._tracks:
            return list(self._tracks[trackId].keys())
        else:
            return []

    def get_feature_locations(self, frameId, trackId):
        if trackId in self._tracks and frameId in self._tracks[trackId]:
            return self._tracks[trackId][frameId]
        else:
            return None
    #
    # def extend_database(self, frameId, matches):
    #     for track_id, feature_location in matches.items():
    #         self.add_track(track_id, frameId, feature_location)

    def serialize(self, file_path):
        data = {
            'tracks': self._tracks,
            'frame_ids': self._frame_ids
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def deserialize(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self._tracks = data['tracks']
            self._frame_ids = data['frame_ids']

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
        self._last_insertions[(frameId + 1) % 2] = {}

    def get_num_tracks(self):
        return self._num_tracks

    def get_num_frames(self):
        return self._num_frames

    def get_mean_track_length(self):
        result = 0

        for key in self._tracks.keys():
            result += len(self._tracks[key])

        self._mean_track_length = result

        try:
            return self._mean_track_length/self._num_tracks

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
            return self._mean_frame_links/self._num_frames

        except ZeroDivisionError:
            return 0

