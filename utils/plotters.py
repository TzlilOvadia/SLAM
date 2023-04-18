import numpy as np
from matplotlib import pyplot as plt


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
    draw_layer(img1, x1_in, x1_out, y1_in, y1_out)

    fig.add_subplot(2, 1, 2)
    draw_layer(img2, x2_in, x2_out, y2_in, y2_out)

    fig.show()


def draw_layer(img, x2_in, x2_out, y2_in, y2_out, s=1):
    plt.imshow(img, cmap='gray')
    plt.scatter(x2_out, y2_out, s=s, color='cyan')
    plt.scatter(x2_in, y2_in, s=s, color='orange')
    plt.yticks([])
    plt.xticks([])


def draw_3d_points(points_list, title='', s=5):
    points_in_3d_np = np.array(points_list)
    xs = points_in_3d_np[:, 0]
    ys = points_in_3d_np[:, 1]
    zs = points_in_3d_np[:, 2]
    print(f"min z after triangulation : {zs.min()}, number of negative zs is {np.where(zs<0)[0].size} out of {len(zs)}")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, c='blue', marker='o', s=s)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)
    ax.set_title(title)
    ax.set_xlim((-100, 100))
    ax.set_ylim((-100, 100))
    ax.set_zlim((-300, 1000))
    ax.invert_yaxis()
    plt.show()
