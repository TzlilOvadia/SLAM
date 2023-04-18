import cv2
import numpy as np
import utils.utils as utils
from models.Matcher import Matcher
from utils.plotters import gen_hist, draw_matches


def calc_y_coord_deviations(y1, y2, thresh=0):
    """

    :param matchs:
    :param thresh:
    :return:
    """

    deviations = np.abs(y1 - y2)
    mask = deviations > thresh
    return deviations[mask].tolist()




def rectified_stereo_classifier(y1: np.ndarray, y2:np.ndarray, thresh=1):
    deviations = np.abs(y1 - y2)
    inliers_idx = np.argwhere(deviations <= thresh)[:,0]
    outliers_idx = np.argwhere(deviations > thresh)[:,0]

    return inliers_idx, outliers_idx



def rectificatied_stereo_pattern(y1, y2):

    inliers_idx, outliers_idx = rectified_stereo_classifier(y1, y2)
    img1in, img2in = indices_mapping[0, inliers_idx], indices_mapping[1, inliers_idx]
    img1out, img2out = indices_mapping[0, outliers_idx], indices_mapping[1, outliers_idx]
    return img1in, img2in, img1out, img2out


def least_squares(p1, p2, Pmat, Qmat):
    x1, y1 = p1
    x2, y2 = p2

    P1 = Pmat[0, :]
    P2 = Pmat[1, :]
    P3 = Pmat[2, :]
    Q1 = Qmat[0, :]
    Q2 = Qmat[1, :]
    Q3 = Qmat[2, :]

    A = np.array([[x1 * P3 - P1],
         [y1 * P3 - P2],
         [x2 * Q3 - Q1],
         [y2 * Q3 - Q2]])

    u, s, vh = np.linalg.svd(A)

    # We saw in class that the solution is located in the last column of vh:
    solution = vh[-1]
    return solution




if __name__ == '__main__':
    img1, img2 = utils.read_images(6)
    h, w = img1.shape

    sift = cv2.SIFT_create()
    matcher = Matcher(sift)
    matcher.detect_and_compute()
    matcher.find_matching_features(with_significance_test=False)
    matches = matcher.get_matches()
    descriptors = matcher.get_dsc()
    kp1, kp2 = matcher.get_kp()

    #### Section 2.1 #####
    x1, y1, x2, y2, indices_mapping = utils.coords_from_kps(matches, kp1, kp2)
    deviations = calc_y_coord_deviations(y1, y2)
    gen_hist(data=deviations, bins=100)

    #### Section 2.2 #####
    # Apply rectificatied stereo pattern on the matches
    img1in, img2in, img1out, img2out = rectificatied_stereo_pattern(y1, y2)

    # Draw matches after using rectified stereo classification test:
    title = "Section 2.2: "
    draw_matches(img1, kp1, img2, kp2, img1in, img2in, img1out, img2out, title=title)

    # Testing the rejection policy under the assumption that the Y-coordinate of erroneous matches is distributed
    # uniformly across the image
    y2_sampled_uniformly = np.random.uniform(0.0, h, len(y2))
    uniform_deviations = calc_y_coord_deviations(y1, y2_sampled_uniformly)
    img1in_uniform, img2in_uniform, img1out_uniform, img2out_uniform = rectificatied_stereo_pattern(y1, y2_sampled_uniformly)
    draw_matches(img1, kp1, img2, kp2, img1in_uniform, img2in_uniform, img1out_uniform, img2out_uniform, title=title)


    #### Section 2.3 #####
    k, m1, m2 = utils.read_cameras()
    p1, p2 = (x1, y1), (x2, y2)
    for pdx in range(len(x1)):
        sol = least_squares((x1[pdx], y1[pdx]), (x2[pdx], y2[pdx]), m1, m2)
        print(sol)

    #### Section 2.4 #####
    # Comparison with cv2 triangulatePoints
    p4d = cv2.triangulatePoints(k @ m1, k @ m2, kp1[0].pt, kp2[0].pt)
    p3d = p4d[:3] / p4d[3]
    print('3D point:', p3d.T)