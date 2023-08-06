import random

import cv2
import numpy as np
from matplotlib import pyplot as plt, cm, colors
from mpl_toolkits.mplot3d import Axes3D


#############################################
################# Constants #################
#############################################

HORIZONTAL_REPRESENTATION = 0
VERTICAL_REPRESENTATION = 1
DEFAULT_POV = 0
TOP_POV = 80
SIDE_POV = -120


def gen_hist(data, bins, title="", x="X", y="Y", path=""):
    """
    :param data:
    :param bins:
    :return:
    """
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    ax.set_title(title)
    plt.ylabel(y)
    plt.xlabel(x)
    if path:
        plt.savefig(f"plots/{path}")
    else:
        plt.show()


def plot_mean_factor_error(initial_errors, optimized_errors, key_frames, path=""):
    plt.figure()
    plt.plot(key_frames, initial_errors, label="Initial Error", color='orange')
    plt.plot(key_frames, optimized_errors, label="Optimized Error", color='blue')
    plt.xlabel("Bundle Starting at Frame Idx")
    plt.ylabel("Mean Factor Error")
    plt.title("Mean Factor Error Before and After Bundle Optimization")
    plt.legend()
    if path:
        plt.savefig(f"plots/{path}")
    else:
        plt.show()


def plot_median_factor_error(initial_errors, optimized_errors, key_frames, path=""):
    plt.figure()
    plt.plot(key_frames, initial_errors, label="Initial Error", color='orange')
    plt.plot(key_frames, optimized_errors, label="Optimized Error", color='blue')
    plt.xlabel("Bundle Starting at Frame Idx")
    plt.ylabel("Median Factor Error")
    plt.title("Median Factor Error Before and After Bundle Optimization")
    plt.legend()
    if path:
        plt.savefig(f"plots/{path}")
    else:
        plt.show()


def plot_loop_between_two_frames(camera_positions, first, second, key_frames, path="plots/lc_"):
    plt.figure()
    plt.scatter(x=camera_positions[:, 0], y=camera_positions[:, 2], color='blue', label='our trajectory', s=0.75)
    plt.scatter(x=camera_positions[first, 0], y=camera_positions[first, 2], color='green', label='first frame', s=20)
    plt.scatter(x=camera_positions[second, 0], y=camera_positions[second, 2], color='red', label='second frame', s=20)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(f"compare frame {key_frames[first]} with {key_frames[second]}")
    plt.legend()
    plt.savefig(path + f"trajectory_{first}_{second}")
    draw_matching_images(first, second, key_frames)


def draw_matching_images(first, second, key_frames, path="lc_compare_images"):
    from utils.utils import read_images
    file_index1, file_index2 = key_frames[first], key_frames[second]
    left1, _ = read_images(file_index1)
    left2, _ = read_images(file_index2)
    img3 = cv2.vconcat([cv2.cvtColor(left1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(left2, cv2.COLOR_GRAY2BGR)])
    plt.figure(figsize=(16, 9))
    plt.imshow(img3)
    plt.title(f"compare image {file_index1} and {file_index2}")
    plt.savefig(f"plots/{path}_{first}_{second}")


def plot_trajectory_with_loops(camera_positions, loop_closures, path="plots/"):
    firsts = [lc[0] for lc in loop_closures]
    seconds = [lc[1] for lc in loop_closures]
    plt.figure()
    plt.scatter(x=camera_positions[:, 0], y=camera_positions[:, 2], color='blue', label='our trajectory', s=0.75)
    plt.scatter(x=camera_positions[firsts, 0], y=camera_positions[firsts, 2], color='green', label='first frame', s=10)
    plt.scatter(x=camera_positions[seconds, 0], y=camera_positions[seconds, 2], color='red', label='second frame', s=10)
    step = 25
    for i in range(0, len(camera_positions[:, 0]), step):
        plt.text(camera_positions[:, 0][i], camera_positions[:, 2][i], str(i), fontsize=6, ha='left', va='bottom')
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(f"our trajectory with all loop closures")
    plt.legend()
    plt.savefig(path + f"traj_with_all_loop_closures")


def plot_uncertainty_over_time(keyframes, uncertainty_score, path="", suffix=""):
    plt.figure()
    plt.plot(keyframes, uncertainty_score)
    plt.xlabel("KeyFrame Index")
    plt.ylabel("uncertainty score per frame")
    plt.title("uncertainty score per frame " + suffix)
    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_localization_error_over_time(keyframes, camera_positions, gt_camera_positions, path="", mode=""):
    plt.figure()
    localization_error = np.linalg.norm(np.abs(camera_positions - gt_camera_positions), axis=1)
    plt.plot(keyframes, localization_error)
    plt.xlabel("KeyFrame Index")
    plt.ylabel("Localization error (meters)")
    plt.title(f"{mode} KeyFrame Localization Error Over Time")
    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_multiple_localization_error_over_time(keyframes, camera_positions_pnp, camera_positions_ba, gt_camera_positions, path="", mode=""):
    plt.figure()
    localization_error_pnp = np.linalg.norm(np.abs(camera_positions_pnp[keyframes] - gt_camera_positions[keyframes]), axis=1)
    localization_error_ba = np.linalg.norm(np.abs(camera_positions_ba - gt_camera_positions[keyframes]), axis=1)
    localization_error_lc = np.linalg.norm(np.abs(camera_positions_lc - gt_camera_positions[keyframes]), axis=1)
    plt.plot(keyframes, localization_error_pnp, label="PNP")
    plt.plot(keyframes, localization_error_ba, label="Bundle Adjustment")
    plt.plot(keyframes, localization_error_lc, label="Loop Closure")
    plt.xlabel("KeyFrame Index")
    plt.ylabel("Localization error (meters)")
    plt.title(f"{mode} KeyFrame Localization Error Over Time")
    plt.legend()
    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_trajectory_and_points(camera_positions, points_locations):
    plt.figure()
    plt.scatter(x=camera_positions[:, 0], y=camera_positions[:, 2], color='blue', label='camera locations', s=0.75)
    plt.scatter(x=points_locations[:, 0], y=points_locations[:, 2], color='orange', label='points locations', s=0.75)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Camera locations and landmarks")
    plt.legend()
    plt.savefig("q2_camera_and_points")





def plot_trajectories(camera_positions, gt_camera_positions, points_3d=None, path="", suffix=""):
    plt.figure()
    plt.scatter(x=camera_positions[:, 0], y=camera_positions[:, 2], color='blue', label='our trajectory', s=0.75)
    #plt.scatter(x=camera_positions[:10, 0], y=camera_positions[:10, 2], color='blue', s=20)
    plt.scatter(x=gt_camera_positions[:, 0], y=gt_camera_positions[:, 2], color='orange', label='ground truth trajectory', s=0.75)
    if points_3d is not None:
        plt.scatter(x=points_3d[:, 0], y=points_3d[:, 2], color='gray', label='landmarks', s=0.2)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Our Trajectory Vs Ground Truth Trajectory "+suffix)
    plt.legend()
    if path:
        plt.savefig(path)
    else:
        plt.show()

def plot_multiple_trajectories(camera_positions_PNP,camera_positions_bundle_adjustment, camera_positions_loop_closure,
                               gt_camera_positions, points_3d=None, path="all_trajectories_combined", suffix=""):

    plt.figure()
    plt.scatter(x=camera_positions_PNP[:, 0], y=camera_positions_PNP[:, 2], color='red', label='PNP trajectory', s=0.5, alpha=.5)
    plt.scatter(x=camera_positions_bundle_adjustment[:, 0], y=camera_positions_bundle_adjustment[:, 2], color='green', label='Bundle adjustment trajectory', s=0.5,alpha=.5)
    plt.scatter(x=camera_positions_loop_closure[:, 0], y=camera_positions_loop_closure[:, 2], color='blue', label='Loop closure trajectory', s=0.5, alpha=.5)

    plt.scatter(x=gt_camera_positions[:, 0], y=gt_camera_positions[:, 2], color='orange', label='Ground truth trajectory', s=0.5,alpha=.5)
    if points_3d is not None:
        plt.scatter(x=points_3d[:, 0], y=points_3d[:, 2], color='gray', label='landmarks', s=0.2)

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Our Trajectory Vs Ground Truth Trajectory "+suffix)
    plt.legend()
    if path:
        plt.savefig(f"plots/{path}")
    else:
        plt.show()


def plot_2d_cameras_and_points(cameras_x, cameras_z, points_x, points_z, path=""):
    plt.figure()
    plt.scatter(x=cameras_x, y=cameras_z, color='blue', label="camera locations")
    plt.scatter(x=points_x, y=points_z, color='orange', label="landmarks")
    plt.title("Camera and Landmarks Locations")
    plt.legend()
    if path:
        plt.savefig(path)
    else:
        plt.show()


def draw_supporting_matches_general(file_index1, file_index2, matcher, matches, supporting_indices):
    """
       Draws supporting matches between key-points in two general images.
       This is done, for sake of consistency, on the left side camera images.

       Parameters:
       - file_index (int): The index of the file/image from KITTI dataset.
       - matcher (object): An object of Matcher class.
       - matches (list): List of matches between keypoints in two consecutive images.
       - supporting_indices (list): List of supporting indices indicating the quality of the matches.

       Returns: None

       Functionality:
       - This function takes the input parameters and performs the following tasks:
           - Loads the left images and their key-points from the matcher object.
           - Concatenates the left images horizontally.
           - Sorts the matches based on their distance and selects the top 150 matches.
           - Draws lines between key-points in the two consecutive images based on the supporting indices.
           - Displays the image with drawn lines in a separate window.
       """
    im1_left = matcher.get_images(file_index1)[0]
    im2_left = matcher.get_images(file_index2)[0]
    kp1_left = matcher.get_kp(file_index1)[0]
    kp2_left = matcher.get_kp(file_index2)[0]
    colors = [(255,0,0), (0,255,0)]
    img3 = cv2.vconcat([cv2.cvtColor(im1_left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(im2_left, cv2.COLOR_GRAY2BGR)])
    # sort_matches = sorted(matches, key=lambda x: x[0].distance)[:min(len(matches), 150)]
    print(len(kp1_left))
    print(len(kp2_left))
    for i, match in enumerate(matches):
        img1_idx = match[0].queryIdx
        img2_idx = match[0].trainIdx
        if img2_idx >= len(kp2_left) or img2_idx>= len(kp1_left):
            continue
        (x1, y1) = kp1_left[img1_idx].pt
        (x2, y2) = kp2_left[img2_idx].pt
        y2 += im1_left.shape[0]  # Shift the second image points down by the height of the first image
        color = colors[int(supporting_indices[i])] if supporting_indices is not None else colors[1]
        # Assign different thickness to the unsupported indices
        thickness = 1 if supporting_indices is None or supporting_indices[i] else 4
        cv2.line(img3, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)

    plt.figure(figsize=(16, 9))
    plt.imshow(img3)
    plt.title("Supporting [Green], Unsupported [Red]" if supporting_indices is not None else "Matches between left_0 to left_1")
    plt.savefig(f"plots/lc_consensus_matches_{file_index1}_{file_index2}")



def draw_supporting_matches(file_index, matcher, matches, supporting_indices):
    """
       Draws supporting matches between key-points in two consecutive images.
       This is done, for sake of consistency, on the left side camera images.

       Parameters:
       - file_index (int): The index of the file/image from KITTI dataset.
       - matcher (object): An object of Matcher class.
       - matches (list): List of matches between keypoints in two consecutive images.
       - supporting_indices (list): List of supporting indices indicating the quality of the matches.

       Returns: None

       Functionality:
       - This function takes the input parameters and performs the following tasks:
           - Loads the left images and their key-points from the matcher object.
           - Concatenates the left images horizontally.
           - Sorts the matches based on their distance and selects the top 150 matches.
           - Draws lines between key-points in the two consecutive images based on the supporting indices.
           - Displays the image with drawn lines in a separate window.
       """
    im1_left = matcher.get_images(file_index - 1)[0]
    im2_left = matcher.get_images(file_index)[0]
    kp1_left = matcher.get_kp(file_index - 1)[0]
    kp2_left = matcher.get_kp(file_index)[0]
    colors = [(255,0,0), (0,255,0)]
    img3 = cv2.vconcat([cv2.cvtColor(im1_left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(im2_left, cv2.COLOR_GRAY2BGR)])
    # sort_matches = sorted(matches, key=lambda x: x[0].distance)[:min(len(matches), 150)]
    for i, match in enumerate(matches):
        img1_idx = match[0].queryIdx
        img2_idx = match[0].trainIdx
        (x1, y1) = kp1_left[img1_idx].pt
        (x2, y2) = kp2_left[img2_idx].pt
        y2 += im1_left.shape[0]  # Shift the second image points down by the height of the first image
        color = colors[int(supporting_indices[i])] if supporting_indices is not None else colors[1]
        # Assign different thickness to the unsupported indices
        thickness = 1 if supporting_indices is None or supporting_indices[i] else 4
        cv2.line(img3, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)

    plt.figure(figsize=(16, 9))
    plt.imshow(img3)
    plt.title("Supporting [Green], Unsupported [Red]" if supporting_indices is not None else "Matches between left_0 to left_1")
    plt.show()



def draw_inlier_and_outlier_matches(img1, kp1, img2, kp2, img1in, img2in, img1out, img2out, title=""):
    """
    Drawing matches with respect to inliers and outliers
    :param img1: image 1
    :param kp1: image 1 key-points
    :param img2: image 2
    :param kp2: image 2 key-points
    :param img1in: image 1 inliers
    :param img2in: image 2 inliers
    :param img1out: image 1 outliers
    :param img2out: image 2 outliers
    :param title:
    :return:
    """
    x1_in, y1_in = np.array([kp1[idx].pt for idx in img1in]).T
    x2_in, y2_in = np.array([kp2[idx].pt for idx in img2in]).T
    x1_out, y1_out = np.array([kp1[idx].pt for idx in img1out]).T
    x2_out, y2_out = np.array([kp2[idx].pt for idx in img2out]).T

    fig = plt.figure()
    fig.suptitle(title)

    fig.add_subplot(2, 1, 1)
    _draw_layer(img1, x1_in, x1_out, y1_in, y1_out)

    fig.add_subplot(2, 1, 2)
    _draw_layer(img2, x2_in, x2_out, y2_in, y2_out)

    fig.show()


def _draw_layer(img, x2_in, x2_out, y2_in, y2_out, s=1):
    """
    :param img:
    :param x2_in:
    :param x2_out:
    :param y2_in:
    :param y2_out:
    :param s:
    :return:
    """
    plt.imshow(img, cmap='gray')
    plt.scatter(x2_out, y2_out, s=s, color='cyan')
    plt.scatter(x2_in, y2_in, s=s, color='orange')
    plt.yticks([])
    plt.xticks([])


def draw_3d_points(points_list: list, title='', s=50, other_points=None, num_points=2000, pov=DEFAULT_POV):
    """
    Creates a 3D point cloud from the provided points list
    :param points_list: List of triangulated points
    :param title: Detailed title of the point cloud
    :param s: size factor for a point representation
    :param other_points: other list of triangulated points (Optional)
    :param num_points: How many points we want to take (Upper bound)
    :param pov: What is the point of view we want
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    num_points = min(num_points, len(points_list), len(other_points) if other_points else len(points_list))
    xs, ys, zs = _trim_and_filter(num_points, points_list)
    if pov == TOP_POV:
        ax.view_init(pov)
    elif pov == SIDE_POV:
        ax.view_init(None, pov)
    size_scaling = _assign_size_for_each_sample(xs, ys, zs, s)
    # Create a scatter plot using scatter method, and set the color based on the z-values
    ax.scatter(xs, ys, zs, c='blue', alpha=.4, marker='o', s=size_scaling, label="Transformed Cloud")
    if other_points is not None:
        xst, yst, zst = _trim_and_filter(num_points, other_points)
        size_scaling_t = _assign_size_for_each_sample(xst, yst, zst, s)
        ax.scatter(xst, yst, zst, c='red', alpha=.4, marker='*', s=size_scaling_t, label="Original Cloud")
        ax.legend(loc='upper left')
    ax.auto_scale_xyz(xs,ys,zs)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)
    ax.set_title(title)
    # Rotate the plot by dragging the mouse
    plt.show()


def _assign_size_for_each_sample(xs, ys, zs, s, obs_pos=np.array([0, 0, 0])):
    """
    Assigns a size scaling factor to each point based on their distance from the observer's position
    :param xs: X-coordinates of the points
    :param ys: Y-coordinates of the points
    :param zs: Z-coordinates of the points
    :param s: Base size factor for a point representation
    :return: Array of size scaling factors for each point
    """
    distances = np.sqrt((xs - obs_pos[0]) ** 2 + (ys - obs_pos[1]) ** 2 + (zs - obs_pos[2]) ** 2)
    # Define the size scaling factor as the inverse of the distance
    size_scaling = s / distances
    return size_scaling


def _trim_and_filter(num_points, points_in_3d):
    """
    Trim and filter the 3D points array based on a specified number of points and percentile values for each axis
    :param num_points: Maximum number of points to take
    :param points_in_3d: List of 3D points
    :return: X, Y, Z coordinates of the trimmed and filtered points
    """
    points_in_3d_np = np.array(points_in_3d)
    xs = points_in_3d_np[:num_points, 0]
    ys = points_in_3d_np[:num_points, 1]
    zs = points_in_3d_np[:num_points, 2]

    # Trimming according to Z's
    xs = xs[(zs < np.percentile(zs, 90)) & (zs > np.percentile(zs, 10))]
    ys = ys[(zs < np.percentile(zs, 90)) & (zs > np.percentile(zs, 10))]
    zs = zs[(zs < np.percentile(zs, 90)) & (zs > np.percentile(zs, 10))]

    # Trimming according to X's
    ys = ys[(xs < np.percentile(xs, 90)) & (xs > np.percentile(xs, 10))]
    zs = zs[(xs < np.percentile(xs, 90)) & (xs > np.percentile(xs, 10))]
    xs = xs[(xs < np.percentile(xs, 90)) & (xs > np.percentile(xs, 10))]

    # Trimming according to Y's
    zs = zs[(ys < np.percentile(ys, 90)) & (ys > np.percentile(ys, 10))]
    xs = xs[(ys < np.percentile(ys, 90)) & (ys > np.percentile(ys, 10))]
    ys = ys[(ys < np.percentile(ys, 90)) & (ys > np.percentile(ys, 10))]

    return xs, ys, zs

def draw_line(image, start_point, end_point, color, thickness, collisions, max_collisions):
    # Calculate alpha value based on number of collisions
    alpha = 1.0 - min(1.0, collisions / float(max_collisions))
    # Adjust color alpha value
    color = (color[0], color[1], color[2], int(alpha * 255))
    # Draw line on image with adjusted color and thickness
    cv2.line(image, start_point, end_point, color, thickness)

def draw_matches(matches, im1, im2, img1_kp, img2_kp, num_of_matches=5000, debug=False, display=VERTICAL_REPRESENTATION):
    """
    This function draws the matches found by the matching algorithm between the two input images.
    :param display:
    :param num_of_matches: Maximum number of matches to be randomly chosen and displayed
    :return: None
    """
    current_num_of_matches = len(matches)
    if current_num_of_matches >= num_of_matches:
        matches = random.sample(matches, num_of_matches)
    print(f"drawing {len(matches)} matches")
    img3 = cv2.drawMatchesKnn(im1, img1_kp, im2, img2_kp, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if display is VERTICAL_REPRESENTATION:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 0) for _ in matches]
        img3 = cv2.vconcat([cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)])
        for i, match in enumerate(matches):
            img1_idx = match[0].queryIdx
            img2_idx = match[0].trainIdx
            (x1, y1) = img1_kp[img1_idx].pt
            (x2, y2) = img2_kp[img2_idx].pt
            y2 += im1.shape[0]  # Shift the second image points down by the height of the first image
            color = colors[i]
            cv2.line(img3, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)

    plt.figure(figsize=(4, 3))
    plt.imshow(img3)
    plt.show()


def plot_four_cameras(Rt, m2):
    R, t = Rt[:, 0:3], Rt[:, 3]
    left_0 = np.array([0, 0, 0])
    right_0 = - m2[:, 3]
    left_1 = -R.T @ t
    right_1 = left_1 - m2[:, 3]
    fig = plt.figure()
    plt.scatter(x=[left_0[0]], y=[left_0[2]], color='blue', label='left_0')
    plt.scatter(x=[right_0[0]], y=[right_0[2]], color=(0.75, 0.75, 1), label='right_0')
    plt.scatter(x=[left_1[0]], y=[left_1[2]], color='red', label='left_1')
    plt.scatter(x=[right_1[0]], y=[right_1[2]], color=(1, 0.75, 0.75), label='right_1')
    plt.title("X and Z locations of 4 cameras in two consecutive frames")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.show()
    a=5


def plot_regions_around_matching_pixels(left, right, x1, y1, x2, y2, frame_index, path=""):
    left, right = left.copy(), right.copy()
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    region_size = 100
    left_height, left_width = left.shape
    right_height, right_width = right.shape

    x1_min, x1_max = max(0, x1 - region_size // 2), min(left_width, x1 + region_size // 2)
    y1_min, y1_max = max(0, y1 - region_size // 2), min(left_height, y1 + region_size // 2)

    x2_min, x2_max = max(0, x2 - region_size // 2), min(right_width, x2 + region_size // 2)
    y2_min, y2_max = max(0, y2 - region_size // 2), min(right_height, y2 + region_size // 2)

    # Extract the regions of interest from both left and right images
    left_region = left[y1_min:y1_max, x1_min:x1_max]
    right_region = right[y2_min:y2_max, x2_min:x2_max]

    # Calculate the relative coordinates within the regions
    x1_rel = x1 - x1_min
    y1_rel = y1 - y1_min
    x2_rel = x2 - x2_min
    y2_rel = y2 - y2_min

    # Mark the feature as a dot in both regions
    dot_size = 3
    left_x_min = max(0, x1_rel - dot_size + 1)
    left_x_max = min(region_size, x1_rel + dot_size)
    left_y_min = max(0, y1_rel - dot_size + 1)
    left_y_max = min(region_size, y1_rel + dot_size)
    left_region[left_y_min:left_y_max, left_x_min:left_x_max] = 0

    right_x_min = max(0, x2_rel - dot_size + 1)
    right_x_max = min(region_size, x2_rel + dot_size)
    right_y_min = max(0, y2_rel - dot_size + 1)
    right_y_max = min(region_size, y2_rel + dot_size)
    right_region[right_y_min:right_y_max, right_x_min:right_x_max] = 0

    # Create a figure with two subplots for left and right regions
    fig, axes = plt.subplots(1, 2, figsize=(4, 5))

    # Plot the left region
    axes[0].imshow(left_region, cmap='gray')
    axes[0].set_title(f'Left Region ({frame_index})')

    # Plot the right region
    axes[1].imshow(right_region, cmap='gray')
    axes[1].set_title(f'Right Region ({frame_index})')

    # Adjust the spacing between subplots
    fig.tight_layout()
    # Display the plot
    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_dict(d, x_title='', y_title='', title='', path=""):
    x = []
    y = []
    for k, v in d.items():
        x.append(k)
        y.append(v)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    if path:
        plt.savefig(f"plots/{path}")
    else:
        plt.show()


def plot_connectivity_graph(frame_num, outgoing_tracks, path=""):
    plt.figure()
    plt.plot(frame_num, outgoing_tracks)
    plt.xlabel("frame")
    plt.ylabel("outgoing tracks")
    plt.title("Connectivity")
    if path:
        plt.savefig(f"plots/{path}")
    else:
        plt.show()


def plot_reprojection_errors(frame_ids, left_errors, right_errors, frame):
    plt.figure()
    plt.plot(frame_ids, left_errors, label='left images error')
    plt.plot(frame_ids, right_errors, label='right images error')
    plt.xlabel("frame id")
    plt.ylabel("reprojection error")
    plt.title(f"reprojection error per image, triangulated from {frame} frame")
    plt.legend()
    plt.show()


def plot_projections_on_images(left_img, right_img, measurement, projected_before_optimization, projected_after_optimization):
    img3 = cv2.vconcat([cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)])
    measured_x_l, measured_x_r, measured_y_l = measurement.uL(), measurement.uR(), measurement.v()
    initial_x_l, initial_x_r, initial_y_l = projected_before_optimization.uL(), projected_before_optimization.uR(), projected_before_optimization.v()
    optimized_x_l, optimized_x_r, optimized_y_l = projected_after_optimization.uL(), projected_after_optimization.uR(), projected_after_optimization.v()
    measured_y_r, initial_y_r, optimized_y_r = measured_y_l + left_img.shape[0], initial_y_l + left_img.shape[0], optimized_y_l + left_img.shape[0]
    plt.figure(figsize=(16, 9))
    plt.imshow(img3)
    plt.scatter([measured_x_l, measured_x_r], [measured_y_l, measured_y_r], s=20, label="measured")
    plt.scatter([initial_x_l, initial_x_r], [initial_y_l, initial_y_r], s=20, label="initial")
    plt.scatter([optimized_x_l, optimized_x_r], [optimized_y_l, optimized_y_r], s=20, label="optimized")
    plt.legend()
    plt.savefig("q2_compare_projections")