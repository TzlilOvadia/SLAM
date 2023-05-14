import random

import cv2
import numpy as np
from matplotlib import pyplot as plt, cm, colors

HORIZONTAL_REPRESENTATION = 0
VERTICAL_REPRESENTATION = 1
def gen_hist(data, bins, title="", x="X", y="Y"):
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
    plt.show()

def draw_supporting_matches(file_index, matcher, consensus_matches, supporting_indices):
    im1, im2 = matcher.get_images(file_index)
    kp1, kp2 = matcher.get_kp(file_index)
    colors = [(255,0,0), (0,255,0)]
    img3 = cv2.hconcat([cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)])
    for i, match in enumerate(consensus_matches):
        img1_idx = match[0]
        img2_idx = match[1]
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        x2 += im1.shape[1]  # Shift the second image points down by the height of the first image
        color = colors[int(supporting_indices[i])]
        cv2.line(img3, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)

    plt.figure(figsize=(4, 3))
    plt.imshow(img3)
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


def draw_3d_points(points_list: list, title='', s=5):
    """
    Creates a 3D point cloud from the provided points list
    :param points_list: points in 3d space to plot
    :param title: An informative title to describe the plot
    :param s: The size of each point in the 3d plot
    :return:
    """
    points_in_3d_np = np.array(points_list)
    xs = points_in_3d_np[:, 0]
    ys = points_in_3d_np[:, 1]
    zs = points_in_3d_np[:, 2]
    print(f"min z after triangulation : {zs.min()}, number of negative zs is {np.where(zs<0)[0].size} out of {len(zs)}")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # color = np.array([1, 0, 0, 0.9])  # RGBA format for red color with alpha value
    # colors = np.tile(color, (len(xs), 1))
    s = s * zs // 5
    s[s == 0] = 1
    cmap = cm.ScalarMappable(norm=colors.Normalize(vmin=min(zs), vmax=max(zs)), cmap='coolwarm')

    # Create a scatter plot using scatter method, and set the color based on the z-values

    scatter = ax.scatter(xs, ys, zs, c=cmap.to_rgba(zs), alpha=0.2, marker='o', s=s)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)
    ax.set_title(title)

    # Create a legend that shows the colormap
    cbar = fig.colorbar(cmap)
    cbar.ax.set_ylabel('Z values')


    ax.set_xlim((-100, 100))
    ax.set_ylim((-100, 100))
    ax.set_zlim((-300, 1000))
    ax.invert_yaxis()
    plt.show()

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
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in matches]
        img3 = cv2.vconcat([cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)])
        for i, match in enumerate(matches):
            img1_idx = match[0].queryIdx
            img2_idx = match[0].trainIdx
            (x1, y1) = img1_kp[img1_idx].pt
            (x2, y2) = img2_kp[img2_idx].pt
            y2 += im1.shape[0]  # Shift the second image points down by the height of the first image
            color = colors[i]
            cv2.line(img3, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)

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