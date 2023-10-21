
"""
ORB: Oriented FAST and Rotated BRIEF
[Dev] 2023.09.27
"""
import numpy as np
import cv2
import os.path as osp
import argparse
from logzero import logger


def createGaussianSigma(nOctaves, sigma):
    gaussianSigma = []
    k = 2 ** (1 / nOctaves)
    gaussianSigma.append(sigma)
    for i in range(1, nOctaves):
        prevSigma = k ** (i - 1) * gaussianSigma[-1]
        totalSigma = k * prevSigma
        curSigma = np.sqrt(totalSigma ** 2 - prevSigma ** 2)
        gaussianSigma.append(curSigma)
    return gaussianSigma


def createGaussianPyramid(image, nOctaves, pyramidSigmas):
    imagePyramid = []
    imagePyramid.append(image)
    for i in range(1, nOctaves):
        curSigma = pyramidSigmas[i]
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=curSigma, sigmaY=curSigma)
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        imagePyramid.append(image)
    imagePyramid = np.array(imagePyramid, dtype=object)
    return imagePyramid


def fastMN(image, octaveIdx, M, N, borderWidth, contrastThreshold):
    keypoints = []
    h, w = image.shape
    scoreMap = np.zeros((h, w))
    if N == 8:
        surPtIdx = np.array([
            [-1, 0], [-1, -1], [0, -1], [1, -1],
            [1, 0], [1, 1], [0, 1], [-1, 1]
        ])
        quickIdx = [0, 2, 4, 6]
    elif N == 12:
        surPtIdx = np.array([
            [-2, 0], [-2, -1], [-1, -2],
            [0, -2], [1, -2], [2, -1],
            [2, 0], [2, 1], [1, 2],
            [0, 2], [-1, 2] ,[-2, 1]
        ])
        quickIdx = [0, 3, 6, 9]
    elif N == 16:
        surPtIdx = np.array([
            [-3, 0], [-3, -1], [-2, -2], [-1, -3],
            [0, -3], [1, -3], [2, -2], [3, -1],
            [3, 0], [3, 1], [2, 2], [1, 3],
            [0, 3], [-1, 3], [-2, 2], [-3, 1]
        ])
        quickIdx = [0, 4, 8, 12]
    else:
        raise ValueError("N must be 8, 12 or 16")
    quickPtIdx = surPtIdx[quickIdx]

    for j in range(borderWidth, h - borderWidth):
        for i in range(borderWidth, w - borderWidth):
            Ip = image[j, i]
            
            if contrastThreshold < 1:
                contrastThreshold = Ip * contrastThreshold
            
            quickRegionPixelVal = image[j + quickPtIdx[:, 1], i + quickPtIdx[:, 0]]
            quickRes1 = np.count_nonzero(Ip + contrastThreshold < quickRegionPixelVal)
            quickRes2 = np.count_nonzero(Ip - contrastThreshold > quickRegionPixelVal)
            if quickRes1 < 3 and quickRes2 < 3:
                continue
            surRegionPixelVal = image[j + surPtIdx[:, 1], i + surPtIdx[:, 0]]
            allRes1 = np.count_nonzero(Ip + contrastThreshold < surRegionPixelVal)
            allRes2 = np.count_nonzero(Ip - contrastThreshold > surRegionPixelVal)
            if allRes1 < M and allRes2 < M:
                continue
            score = np.sum(np.abs(Ip - surRegionPixelVal))
            scoreMap[j, i] = score

            keypoint = cv2.KeyPoint()
            keypoint.pt = (i * 2 ** (octaveIdx), j * 2 ** (octaveIdx))
            keypoints.append(keypoint)
    return keypoints, scoreMap


def NonMaximumSuppression(scoreMap, octaveIdx, nmsWinSize, borderWidth):
    h, w = scoreMap.shape
    winRadius = nmsWinSize // 2
    keypoints = []
    for j in range(h):
        if (j - winRadius <= 0) or (j + winRadius + 1 >= h - 1):
            continue
        for i in range(w):
            if (i - winRadius <= 0) or (i + winRadius + 1 >= w - 1):
                continue
            if scoreMap[j, i] == 0:
                continue
            scoreRegion = scoreMap[j - winRadius : j + winRadius + 1, i - winRadius : i + winRadius + 1]
            maxScoreIdx = np.argmax(scoreRegion)
            maxScoreLoc = np.unravel_index(maxScoreIdx, scoreRegion.shape)
            maxScoreX = i - winRadius + maxScoreLoc[1]
            maxScoreY = j - winRadius + maxScoreLoc[0]

            if (maxScoreX != i) or (maxScoreY != j):
                scoreMap[j, i] = 0
                continue
            keypoint = cv2.KeyPoint()
            keypoint.pt = (i * 2 ** octaveIdx, j * 2 ** octaveIdx)
            keypoints.append(keypoint)
    return keypoints, scoreMap


def calcKeypointOrientation(keypoint, image, radius, octaveIdx):
    x, y = keypoint.pt
    x, y = x / (2 ** octaveIdx), y / (2 ** octaveIdx) 
    x, y = int(x), int(y)

    imageRegion = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
    h_region, w_region = imageRegion.shape
    m00 = np.sum(imageRegion)
    m10, m01 = 0, 0
    for j in range(h_region):
        for i in range(w_region):
            m10 += i * imageRegion[j, i]
            m01 += j * imageRegion[j, i]
    
    cx = m10 / (m00 + 1e-7)
    cy = m01 / (m00 + 1e-7)
    if cx < 0:
        cx = 0
    elif cx >= h_region:
        cx = h_region - 1
    if cy < 0:
        cy = 0
    elif cy >= w_region:
        cy = w_region - 1
    
    orientation = np.rad2deg(np.arctan2(m01, m10))
    newKeypoint = cv2.KeyPoint()
    newKeypoint.pt = keypoint.pt
    newKeypoint.angle = orientation
    return newKeypoint


def rBRIEF(image, keypoint, descDim, patchSize, octaveIdx, randomSeed=42):
    random = np.random.RandomState(seed=randomSeed)
    h, w = image.shape

    samples = np.array((patchSize / 5) * random.randn(descDim * 8), dtype=np.int8)
    mask1 = samples < (patchSize // 2)
    mask2 = samples > (- (patchSize - 2) // 2)
    samples = samples[mask1 & mask2]
    coord1 = samples[: descDim * 2].reshape(descDim, 2)
    coord2 = samples[descDim * 2 : descDim * 4].reshape(descDim, 2)
    descriptor = np.zeros((descDim), dtype=np.int8)

    for pIdx in range(coord1.shape[0]):
        pr0 = coord1[pIdx, 0]
        pc0 = coord1[pIdx, 1]
        pr1 = coord2[pIdx, 0]
        pc1 = coord2[pIdx, 1]

        angle = keypoint.angle
        cosTheta = np.cos(np.deg2rad(angle))
        sinTheta = np.sin(np.deg2rad(angle))
        kr = int(np.round(keypoint.pt[1]))
        kc = int(np.round(keypoint.pt[0]))
        rotation_kr = kc * sinTheta + kr * cosTheta
        rotation_kc = kc * cosTheta - kr * sinTheta
        rotation_kr = int(np.round(rotation_kr))
        rotation_kc = int(np.round(rotation_kc))

        if rotation_kr + pr0 <= 0 or rotation_kr + pr0 >= h - 1 or rotation_kc + pc0 <=0 or rotation_kc + pc0 >= w - 1:
            Ip1 = image[kr + pr0, kc + pc0]
        else:
            Ip1 = image[rotation_kr + pr0, rotation_kc + pc0]
        
        if rotation_kr + pr1 <= 0 or rotation_kr + pr1 >= h - 1 or rotation_kc + pc1 <=0 or rotation_kc + pc1 >= w - 1:
            Ip2 = image[kr + pr1, kc + pc1]
        else:
            Ip2 = image[rotation_kr + pr1, rotation_kc + pc1]
    
        if Ip1 < Ip2:
            descriptor[pIdx] = True
    return descriptor


def ORB(image, nOctaves, sigma, fastM, fastN, contrastThreshold,
        borderWidth, nmsWinSize, descDim, descPatchSize, 
        plotFlag, savePath
):
    originImage = image.copy()
    initSigma = np.sqrt(sigma * sigma - 0.5 * 0.5)
    baseImage = cv2.GaussianBlur(image, (0, 0), sigmaX=initSigma, sigmaY=initSigma)
    h, w = baseImage.shape
    logger.debug(f'Base image shape: {h} * {w}')
    
    gaussianSigmas = createGaussianSigma(nOctaves, sigma)
    logger.debug(f'Gaussian sigmas: {gaussianSigmas}')

    imagePyramid = createGaussianPyramid(image, nOctaves, gaussianSigmas)
    logger.debug(f'Image pyramid shape: {imagePyramid.shape}')

    keypointsWithOrientaion = []
    descriptors = []
    for octaveIdx in range(nOctaves):
        curImage = imagePyramid[octaveIdx]
        keypoints, scoreMap = fastMN(curImage, octaveIdx, fastM, fastN, borderWidth, contrastThreshold)
        keypoints, scoreMap = NonMaximumSuppression(scoreMap, octaveIdx, nmsWinSize, borderWidth)

        for keypoint in keypoints:
            newKeypoint = calcKeypointOrientation(keypoint, curImage, radius=1, octaveIdx=octaveIdx)
            descriptor = rBRIEF(image, keypoint, descDim, descPatchSize, octaveIdx, randomSeed=42)
            keypointsWithOrientaion.append(newKeypoint)
            descriptors.append(descriptor)

        if plotFlag:
            plotImage = cv2.cvtColor(originImage, cv2.COLOR_GRAY2BGR)
            for keypoint in keypointsWithOrientaion:
                x, y = keypoint.pt
                x, y = int(x), int(y)
                cv2.circle(plotImage, (x, y), radius=2, color=(0, 255, 9), thickness=1, lineType=cv2.LINE_AA)
            cv2.imwrite(osp.join(savePath, f'kpt_orientation/octave_{octaveIdx}.png'), plotImage)

    keypointsWithOrientaion = np.array(keypointsWithOrientaion, dtype=object)
    descriptors = np.array(descriptors, dtype=object)
    logger.debug(f'Keypoints shape: {keypointsWithOrientaion.shape}')
    logger.debug(f'Descriptors shape: {descriptors.shape}')
    
    return keypointsWithOrientaion, descriptors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------------------- IMAGE PATH -------------------------------- #
    parser.add_argument('--imPath', default='test_data/keypoint_occupy_tiny/i/i_books/3.png')
    # -------------------------------- CONFIGS -------------------------------- #
    parser.add_argument('--nOctaves', default=5)
    parser.add_argument('--sigma', default=0.8)
    parser.add_argument('--fastM', default=9, choices=[5, 7, 9])
    parser.add_argument('--fastN', default=16, choices=[8, 12, 16])
    parser.add_argument('--contrastThreshold', default=0.4)
    parser.add_argument('--nmsWinSize', default=6)
    parser.add_argument('--borderWidth', default=5)
    parser.add_argument('--descDim', default=256)
    parser.add_argument('--descPatchSize', default=9)
    # -------------------------------- PLOT IMAGE -------------------------------- #
    parser.add_argument('--plotFlag', default=False)
    parser.add_argument('--plotSaveRoot', default='plots/orb_plots')
    args = parser.parse_args()

    image = cv2.imread(args.imPath, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = ORB(image, nOctaves=args.nOctaves, sigma=args.sigma, 
                                fastM=args.fastM, fastN=args.fastN, 
                                contrastThreshold=args.contrastThreshold,
                                borderWidth=args.borderWidth, nmsWinSize=args.nmsWinSize,
                                descDim=args.descDim, descPatchSize=args.descPatchSize,
                                plotFlag=args.plotFlag, savePath=args.plotSaveRoot)
    print('Done')