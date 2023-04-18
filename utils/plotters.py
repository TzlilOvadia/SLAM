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