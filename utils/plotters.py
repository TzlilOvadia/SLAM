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


def draw_matches(img1, kp1, img2, kp2, img1in, img2in, img1out, img2out, title=""):
    x1_in, y1_in = np.array([kp1[idx].pt for idx in img1in], dtype=np.int).T
    x2_in, y2_in = np.array([kp2[idx].pt for idx in img2in], dtype=np.int).T
    x1_out, y1_out = np.array([kp1[idx].pt for idx in img1out], dtype=np.int).T
    x2_out, y2_out = np.array([kp2[idx].pt for idx in img2out], dtype=np.int).T

    fig = plt.figure()
    fig.suptitle(title)

    fig.add_subplot(2, 1, 1)
    draw_layer(img1, x1_in, x1_out, y1_in, y1_out)

    fig.add_subplot(2, 1, 2)
    draw_layer(img2, x2_in, x2_out, y2_in, y2_out)

    fig.show()


def draw_layer(img2, x2_in, x2_out, y2_in, y2_out, s=1):
    plt.imshow(img2, cmap='gray')
    plt.scatter(x2_in, y2_in, s=s, color='cyan')
    plt.scatter(x2_out, y2_out, s=s, color='orange')
    plt.yticks([])
    plt.xticks([])



