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

    if debug:
        im_1_pt = img1_kp[212]
        im_2_pt = img2_kp[135]
        im1 = cv2.circle(im1, (np.int(im_1_pt.pt[0]), np.int(im_1_pt.pt[1])), 10, 3, thickness=3, lineType=8,
                         shift=0)
        im2 = cv2.circle(im2, (np.int(im_2_pt.pt[0]), np.int(im_2_pt.pt[1])), 10, 3, thickness=3, lineType=8,
                         shift=0)
    img3 = cv2.drawMatchesKnn(im1, img1_kp, im2, img2_kp, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if display is VERTICAL_REPRESENTATION:
        img3=cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)
    plt.figure(figsize=(4, 3))
    plt.imshow(img3)
    plt.show()