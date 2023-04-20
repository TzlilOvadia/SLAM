import cv2
import numpy as np
import random
import utils.utils as utils
from models.Matcher import Matcher
from utils.plotters import gen_hist, draw_inlier_and_outlier_matches, draw_3d_points


def calc_y_coord_deviations(y1, y2):
    """
    Given y1 and y2, we calculate for each pair of matched points the deviation w.r.t their deviation.
    :param y1: The y-coordinate of all kp1.pt
    :param y2: The y-coordinate of the corresponding y1 (matched by taking the trainIdx)
    :param thresh: Threshold value to determine how to reject matching points
    :return:
    """
    deviations = np.abs(y1 - y2)
    return deviations


def find_median_dist(list1, list2):
    """
    finds the median value of the norm between the corresponding 3d points in list1 and list2.
    :param list1: list of 3d points (Left image)
    :param list2: list of 3d points of img2 (Right image)
    :return:
    """
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    dist = np.linalg.norm(arr1 - arr2, axis=1)
    return np.median(dist)


def _rectified_stereo_classifier(y1: np.ndarray, y2:np.ndarray, thresh=1):
    """
    An helper function for rectificatied_stereo_pattern.
    :param y1: The y-coordinate of all kp1.pt
    :param y2: The y-coordinate of the corresponding y1 (matched by taking the trainIdx)
    :param thresh: Threshold value to determine how to reject matching points
    :return: inliers and outliers nd arrays
    """
    deviations = np.abs(y1 - y2)
    inliers_idx = np.argwhere(deviations <= thresh)[:,0]
    outliers_idx = np.argwhere(deviations > thresh)[:,0]
    return inliers_idx, outliers_idx



def rectificatied_stereo_pattern(y1, y2, thresh=1):
    """
    For each pair of points, this function will return a classification of points to two groups: Inliers and Outliers.
    This is done for each image.
    :param y1: The y-coordinate of all kp1.pt (Left image of the KITTI dataset)
    :param y2: The y-coordinate of the corresponding y1, matched by taking the trainIdx (Right image of the KITTI dataset).
    :param thresh: Threshold value to determine how to reject matching points
    :return:
    """
    inliers_idx, outliers_idx = _rectified_stereo_classifier(y1, y2, thresh)
    img1in, img2in = indices_mapping[0, inliers_idx], indices_mapping[1, inliers_idx]
    img1out, img2out = indices_mapping[0, outliers_idx], indices_mapping[1, outliers_idx]
    return img1in, img2in, img1out, img2out


def least_squares(p1, p2, Pmat, Qmat):
    """
    Least Squares algorithm as learned in class
    :param p1: (x,y) Point in the left image
    :param p2: (x,y) Point in the right image
    :param Pmat: Left image matrix
    :param Qmat: Right image matrix
    :return:
    """
    x1, y1 = p1
    x2, y2 = p2

    P1 = Pmat[0, :]
    P2 = Pmat[1, :]
    P3 = Pmat[2, :]
    Q1 = Qmat[0, :]
    Q2 = Qmat[1, :]
    Q3 = Qmat[2, :]

    # Computing the 'A' matrix from the solution of Ax=0
    A = np.array([x1 * P3 - P1,
         y1 * P3 - P2,
         x2 * Q3 - Q1,
         y2 * Q3 - Q2])

    # Compute the SVD solution for A matrix
    u, s, vh = np.linalg.svd(A)

    # We saw in class that the solution is located in the last column of vh:
    solution_4d = vh[-1]

    # Transform from 4d to 3d:
    solution_3d = solution_4d[:3] / solution_4d[-1]
    return solution_3d



if __name__ == '__main__':

    random.seed(6)
    img1, img2 = utils.read_images(0)
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
    gen_hist(data=deviations, bins=100, x="deviation from rectified stereo pattern", y="Number of matches")
    percentage_of_matches_with_high_deviation = np.where(deviations > 2)[0].size / deviations.size
    print(f"The percentage of matches with deviation higher than 2 is: {percentage_of_matches_with_high_deviation}")

    #### Section 2.2 #####
    # Apply rectified stereo pattern on the matches
    img1in, img2in, img1out, img2out = rectificatied_stereo_pattern(y1, y2, thresh=1)
    # Draw matches after using rectified stereo classification test:
    title = "Section 2.2: "
    draw_inlier_and_outlier_matches(img1, kp1, img2, kp2, img1in, img2in, img1out, img2out, title=title)
    print(f"out of {len(matches)} matches, {len(img1out)} were discarded as outliers")

    #### Section 2.3 #####
    x1_in, y1_in = np.array([kp1[idx].pt for idx in img1in]).T
    x2_in, y2_in = np.array([kp2[idx].pt for idx in img2in]).T
    k, m1, m2 = utils.read_cameras()
    p1, p2 = (x1_in, y1_in), (x2_in, y2_in)
    our_inlier_points_in_3d = []
    cv_inlier_points_in_3d = []

    for idx in range(len(x1_in)):
        our_sol = least_squares((x1_in[idx], y1_in[idx]), (x2_in[idx], y2_in[idx]), k @ m1, k @ m2)
        our_inlier_points_in_3d.append(our_sol)
        cv_p4d = cv2.triangulatePoints(k @ m1, k @ m2, (x1_in[idx], y1_in[idx]), (x2_in[idx], y2_in[idx]))
        cv_sol = (cv_p4d[:3] / cv_p4d[3]).reshape(-1)
        cv_inlier_points_in_3d.append(cv_sol)
    draw_3d_points(our_inlier_points_in_3d, title=f"our 3d points from triangulation from image 0")
    draw_3d_points(cv_inlier_points_in_3d, title=f"cv2 3d points from triangulation from image 0")
    med = find_median_dist(our_inlier_points_in_3d, cv_inlier_points_in_3d)
    print(f"the median distance between points triangulated by us to points triangulated by cv2 is {med}, i.e very small...")

    #### Section 2.4 #####
    for idx in range(0,100,10):
        sift = cv2.SIFT_create()
        matcher = Matcher(sift, file_index=idx)
        matcher.detect_and_compute()
        matcher.find_matching_features(with_significance_test=False)
        matches = matcher.get_matches()
        descriptors = matcher.get_dsc()
        kp1, kp2 = matcher.get_kp()
        x1, y1, x2, y2, indices_mapping = utils.coords_from_kps(matches, kp1, kp2)
        img1in, img2in, img1out, img2out = rectificatied_stereo_pattern(y1, y2, thresh=1)
        x1_in, y1_in = np.array([kp1[idx].pt for idx in img1in]).T
        x2_in, y2_in = np.array([kp2[idx].pt for idx in img2in]).T
        print(f"Image {idx}: out of {len(matches)} matches, {len(img1out)} were discarded as outliers and {len(img1in)} are inliers")
        our_inlier_points_in_3d = [least_squares((x1_in[ind], y1_in[ind]), (x2_in[ind], y2_in[ind]), k @ m1, k @ m2) for ind in range(len(x1_in))]
        draw_3d_points(our_inlier_points_in_3d, title=f"our 3d points from triangulation from image {idx}", s=0.5)