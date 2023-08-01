from numpy import pi
#############################################
################# Constants #################
#############################################
KPS = 0
DSC = 2
HORIZONTAL_REPRESENTATION = 0
VERTICAL_REPRESENTATION = 1
DEFAULT_POV = 0
TOP_POV = 80
SIDE_POV = -120
MATCHES = 2
LEFT = 0
RIGHT = 1
CONSECUTIVE = 3
FRAMES = 4
SUCCESS = 0
FAILURE = 1
LOCATIONS_IDX = 1
LAST_ITEM = -1
FRAME_ID = -1
ROTATION_TOLERANCE = .2
DISTANCE_TOLERANCE = 60
MIN_WINDOW_SIZE = 5
MAX_DISTANCE_TOLERANCE = 150
PERCENTILE = .97
LARGE_ROTATION_VAR = (30 * pi / 180) ** 2
SMALL_ROTATION_VAR = (5 * pi / 180) ** 2
HORIZONTAL_SHIFT_TOLERANCE = 50
VERTICAL_SHIFT_TOLERANCE = 15
CAMERA = "c"
POINT = "q"


I_INDEX = 0
BUNDLE_WINDOW_INDEX = 1
BUNDLE_GRAPH_INDEX = 2
INITIAL_ESTIMATES_INDEX = 3
LANDMARKS_INDEX = 4
OPTIMIZED_ESTIMATES_INDEX = 5
MARGINALS_INDEX = 6
MAHALANOBIS_THRESH = 400000

PATH_TO_SAVE_TRACKER_FILE = "models/serialized_tracker"
PATH_TO_SAVE_BUNDLE_ADJUSTMENT_RESULTS = "models/bundle_adjustment_results"

PATH_TO_SAVE_COMPARISON_TO_GT_PNP = "plots/compare_PNP_to_ground_truth"
PATH_TO_SAVE_COMPARISON_TO_GT_BUNDLE_ADJUSTMENT = "plots/compare_BUNDLE_ADJUSTMENT_to_ground_truth"
PATH_TO_SAVE_COMPARISON_TO_GT_LOOP_CLOSURE = "plots/compare_LOOP_CLOSURE_to_ground_truth"

PATH_TO_SAVE_LOCALIZATION_ERROR_PNP = "plots/localization_error_PNP"
PATH_TO_SAVE_2D_TRAJECTORY_PNP = "plots/2d_view_of_the_entire_scene_PNP"

PATH_TO_SAVE_LOCALIZATION_ERROR_BUNDLE_ADJUSTMENT = "plots/localization_error_BUNDLE_ADJUSTMENT"
PATH_TO_SAVE_2D_TRAJECTORY_BUNDLE_ADJUSTMENT = "plots/2d_view_of_the_entire_scene_BUNDLE_ADJUSTMENT"
PATH_TO_SAVE_COMPARISON_TO_GT = "plots/compare_to_ground_truth"

PATH_TO_SAVE_LOCALIZATION_ERROR_LOOP_CLOSURE_AFTER = "plots/localization_error_LOOP_CLOSURE_AFTER"
PATH_TO_SAVE_LOCALIZATION_ERROR_LOOP_CLOSURE_BEFORE = "plots/localization_error_LOOP_CLOSURE_BEFORE"
PATH_TO_SAVE_2D_TRAJECTORY_LOOP_CLOSURE = "plots/2d_view_of_the_entire_scene_LOOP_CLOSURE"


