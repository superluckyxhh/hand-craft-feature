import numpy as np
import os.path as osp
import cv2
import argparse
import matplotlib.pyplot as plt
from utils import plot_im, plot_points_in_gray, plot_color_points_in_gray


def createGaussianKernel2D(ksize, sigma):
    radius = ksize // 2
    x = np.linspace(-radius, radius, ksize)
    y = np.linspace(-radius, radius, ksize)
    xx, yy = np.meshgrid(x, y)
    dst = xx ** 2 + yy ** 2
    sigma_power = sigma ** 2
    kernel = np.exp(-0.5 * (dst / sigma_power)) / (2 * np.pi * sigma_power)
    kernel = kernel / np.sum(kernel)
    return kernel


def nonMaximalSuppression(window_size, keypoints, score_map):
    """
    Given one keypoint, make wd * wd window to select the max score keypoint  
    """
    win_radius = window_size // 2
    nms_keypoints, nms_scores = [], []
    kpts_num = keypoints.shape[0]
    sH, sW = score_map.shape
    
    for num in range(kpts_num):
        x, y = keypoints[num]
        kpt_window = score_map[max(0, y-win_radius):min(sH, y+win_radius+1), max(0, x-win_radius):min(sW, x+win_radius+1)]
        index = np.argmax(kpt_window)
        loc = np.unravel_index(index, kpt_window.shape)
        loc_x = (x - win_radius) + loc[1]
        loc_y = (y - win_radius) + loc[0]

        nms_kpt = [loc_x, loc_y]
        nms_score = score_map[loc_y, loc_x] 

        if nms_kpt not in nms_keypoints:
            nms_keypoints.append(nms_kpt)
            nms_scores.append(nms_score)

    nms_keypoints = np.array(nms_keypoints)
    nms_scores = np.array(nms_scores)
    return nms_keypoints, nms_scores


def calcKeypointAndScore(im, cpt, sr_pts_ids, q_pts_ids, judge_q_num, judge_sr_num, thd):
    """
    Args:
        im: gray scale image, [H, W]
        cpt: center point, [row, col]
        sr_pts_num: circle surround points number
        thd: score threshold
    Return:
        keypoint_flag: flag = 1 (Keypoint)
        score_map: shape [H, W] 
    
    """
    kpt_flag = 0
    rp, cp = cpt[0], cpt[1]
    Ip = im[rp, cp]
    thd = thd * Ip if thd < 1 else thd

    # Jugde the candidate keypoint quickly
    q_num1 = np.count_nonzero(Ip + thd < im[rp + q_pts_ids[:, 0], cp + q_pts_ids[:, 1]])
    q_num2 = np.count_nonzero(Ip - thd > im[rp + q_pts_ids[:, 0], cp + q_pts_ids[:, 1]])
    if q_num1 >= judge_q_num or q_num2 >= judge_q_num:
        o_num1 = np.count_nonzero(Ip + thd <= im[rp + sr_pts_ids[:, 0], cp + sr_pts_ids[:, 1]])
        o_num2 = np.count_nonzero(Ip - thd >= im[rp + sr_pts_ids[:, 0], cp + sr_pts_ids[:, 1]])
        # Jugde the candidate corner keypoint
        if o_num1 >= judge_sr_num or o_num2 >= judge_sr_num:
            kpt_flag = 1
            score = np.sum(np.abs(Ip - im[sr_pts_ids[:, 0], sr_pts_ids[:, 1]]))
    
    if kpt_flag == 1:
        return kpt_flag, score
    else:
        return kpt_flag, 0


def FAST(image, gaussian_ksize, gaussian_sigma, border_width, numSurroundPoints, numQucikJudgePoints, numJudegSurroundPoints, pixelThreshold, nmsSize):
    keypoints = []
    plot_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    H, W = image.shape
    gaussian_kernel = createGaussianKernel2D(gaussian_ksize, gaussian_sigma)
    image = cv2.filter2D(image, -1, gaussian_kernel)
    
    # Compute the circle keypoints
    surround_point_ids = np.array([
        [0, 3],  [1, 3],   [2, 2],   [3, 1], 
        [3, 0],  [3, -1],  [2, -2],  [1, -3],
        [0, -3], [-1, -3], [-2, -2], [-3, -1],
        [-3, 0], [-3, 1],  [-2, 2],  [-1, 3],
    ])
    # quick circle ids: [0, 4, 8, 12]
    quick_thetas_ids = np.arange(0, numSurroundPoints, 4)
    quick_point_ids = surround_point_ids[quick_thetas_ids]
    scores_map = np.zeros((H, W), dtype=np.float16)
    
    for y in range(border_width, H - border_width):
        for x in range(border_width, W - border_width):
            center_point = [y, x]
            detect_flag, score = calcKeypointAndScore(image, center_point, surround_point_ids, quick_point_ids, numQucikJudgePoints, numJudegSurroundPoints, pixelThreshold)
            if detect_flag == 0:
                continue
            keypoints.append([x, y])
            scores_map[y, x] = score
    keypoints = np.array(keypoints)
    # NMS
    nms_keypoints, nms_scores = nonMaximalSuppression(nmsSize, keypoints, scores_map)
    """ TEST PLOT """
    # for keypoint in nms_keypoints:
    #     cv2.circle(plot_image, keypoint, radius=2, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    # cv2.imwrite('plots/fast_plots/nms.png', plot_image)
    return nms_keypoints, nms_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------------------- IMAGE PATH -------------------------------- #
    parser.add_argument('--im_path', default='test_data/feature_match_data/v/v_bip/1.png')
    # -------------------------------- FAST COMFIGS -------------------------------- #
    parser.add_argument('--numSurroundPoints', default=16)
    parser.add_argument('--borderWidth', default=5)
    parser.add_argument('--baseSigma', default=0.3)
    parser.add_argument('--gaussianKernel', default=5)
    parser.add_argument('--numQucikJudgePoints', default=3)
    parser.add_argument('--numJudegSurroundPoints', default=9)
    parser.add_argument('--pixelThreshold', default=0.4)
    parser.add_argument('--nmsWindowSize', default=5)
    # -------------------------------- PLOT IMAGE -------------------------------- #
    parser.add_argument('--plot_flag', default=True)
    parser.add_argument('--plot_save_root', default='plots/fast_plots')
    # -------------------------------- SAVE FILE -------------------------------- #
    parser.add_argument('--save_kpts_txt_flag', default=False)
    parser.add_argument('--save_kpts_npy_flag', default=False)
    parser.add_argument('--save_kpts_path', default='savefiles/fast_saves')
    args = parser.parse_args()

    if not osp.exists(args.im_path):
        raise ValueError("Invalid image path")
    image = cv2.imread(args.im_path, cv2.IMREAD_GRAYSCALE)    
    keypoints, scores = FAST(image, args.gaussianKernel, args.baseSigma, args.borderWidth , args.numSurroundPoints,
                             args.numQucikJudgePoints, args.numJudegSurroundPoints, args.pixelThreshold, args.nmsWindowSize)
    
    print("Done")