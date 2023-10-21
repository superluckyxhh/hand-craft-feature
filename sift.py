"""
[Chg, 2023.09.25]:
1. 拆分出升采样
2. 利用自定值计算边长
3. 增加移除多余特征点的操作以及 NMS 操作
"""
import numpy as np
import cv2
import os.path as osp
import argparse
import functools
from logzero import logger


def createGaussianSigma(sigma=1.6, nOctaveLayers=3):
    """
        Eq: k ** r * sigma, r in [0, nOctaveLayers + 2]
    """
    S = nOctaveLayers + 3
    gaussianSigmaList = [0] * S
    gaussianSigmaList[0] = sigma
    k = 2 ** (1 / nOctaveLayers)

    for i in range(1, S):
        prevSigma = k ** (i - 1) * sigma
        totalSigma = k * prevSigma
        curSigma = np.sqrt(totalSigma ** 2 - prevSigma ** 2)
        gaussianSigmaList[i] = curSigma
    
    return gaussianSigmaList


def createGaussianPyramid(image, nOctaves, gaussianSigmaList):
    gaussianPyramid = []
    for octaveIdx in range(nOctaves):
        gaussianImagesInOctave = []
        gaussianImagesInOctave.append(image)
        for gaussianSigma in gaussianSigmaList[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussianSigma, sigmaY=gaussianSigma)
            gaussianImagesInOctave.append(image)
        gaussianPyramid.append(gaussianImagesInOctave)

        # Prepare next layer image
        nextLayerImage = gaussianImagesInOctave[-3]
        next_h, next_w = nextLayerImage.shape[0] // 2, nextLayerImage.shape[1] // 2
        image = cv2.resize(nextLayerImage, (next_w, next_h), interpolation=cv2.INTER_LINEAR)

    gaussianPyramid = np.array(gaussianPyramid, dtype=object)
    return gaussianPyramid


def calcDoGPyramid(gaussianPyramid):
    DoGPyramid = []
    for gaussianImagesInOctave in gaussianPyramid:
        DoGImagesInOctave = []
        for firstImage, secondImage in zip(gaussianImagesInOctave, gaussianImagesInOctave[1:]):
            DoGImage = cv2.subtract(secondImage, firstImage)
            DoGImagesInOctave.append(DoGImage)
        DoGPyramid.append(DoGImagesInOctave)
    
    DoGPyramid = np.array(DoGPyramid, dtype=object)
    return DoGPyramid


def judgeExtremaPointInDiscreteImage(firstRegion, secondRegion, thirdRegion, threshold):
    pixelValue = secondRegion[1, 1]
    if np.abs(pixelValue) > threshold:
        if pixelValue > 0:
            return np.all(pixelValue >= firstRegion) and np.all(pixelValue >= thirdRegion) and \
                np.all(pixelValue >= secondRegion[0, :]) and np.all(pixelValue >= secondRegion[2, :]) and \
                (pixelValue >= secondRegion[1, 0]) and (pixelValue >= secondRegion[1, 2])
        
        elif pixelValue < 0:
            return np.all(pixelValue <= firstRegion) and np.all(pixelValue <= thirdRegion) and \
                np.all(pixelValue <= secondRegion[0, :]) and np.all(pixelValue <= secondRegion[2, :]) and \
                (pixelValue <= secondRegion[1, 0]) and (pixelValue <= secondRegion[1, 2])
    return False


def calcPointViaQuadraticFit(j, i, octaveLayerIdx, firstImage, secondImage, thirdImage, sigma, octaveIdx, nOctaveLayers, contrastThreshold, edgeThreshold, iterations, UpSampleFlag, borderWidth):
    h, w = firstImage.shape
    
    for iter in range(iterations):
        firstRegion = firstImage[j - 1 : j + 2, i - 1 : i + 2]
        secondRegion = secondImage[j - 1 : j + 2, i - 1 : i + 2]
        thirdRegion = thirdImage[j - 1 : j + 2, i - 1 : i + 2]
        # RegionCube [Sigma, Y, X]
        RegionCube = np.stack([firstRegion, secondRegion, thirdRegion]) / 255.

        # Calculate the gradient
        dx = 0.5 * (RegionCube[1, 1, 2] - RegionCube[1, 1, 0])
        dy = 0.5 * (RegionCube[1, 2, 1] - RegionCube[1, 0, 1])
        ds = 0.5 * (RegionCube[2, 1, 1] - RegionCube[0, 1, 1])
        gradient = np.array([dx, dy, ds])

        # Calculate the Hessian matrix
        dxx = RegionCube[1, 1, 2] + RegionCube[1, 1, 0] - 2 * RegionCube[1, 1, 1]
        dyy = RegionCube[1, 2, 1] + RegionCube[1, 0, 1] - 2 * RegionCube[1, 1, 1]
        dss = RegionCube[2, 1, 1] + RegionCube[0, 1, 1] - 2 * RegionCube[1, 1, 1]
        dxy = 0.25 * (RegionCube[1, 2, 2] - RegionCube[1, 2, 0] - RegionCube[1, 0, 2] + RegionCube[1, 0, 0])
        dxs = 0.25 * (RegionCube[2, 1, 2] - RegionCube[2, 1, 0] - RegionCube[0, 1, 2] + RegionCube[0, 1, 0])
        dys = 0.25 * (RegionCube[2, 2, 1] - RegionCube[2, 0, 1] - RegionCube[0, 2, 1] + RegionCube[0, 0, 1])
        hessian_matrix = np.array([[dxx, dxy, dxs],
                                [dxy, dyy, dys],
                                [dxs, dys, dss]])
        
        update_list = -1 * np.linalg.lstsq(hessian_matrix, gradient, rcond=None)[0]
        update_x, update_y, update_s = update_list
        if np.abs(update_x) <= 0.5 and np.abs(update_y) <= 0.5 and np.abs(update_s) <= 0.5:
            break

        j += int(np.round(update_y))
        i += int(np.round(update_x))
        octaveLayerIdx += int(np.round(update_s))

        # TODO: if the update value exceeds the boundary, stop
        if i >= w - borderWidth or i < borderWidth or j >= h - borderWidth or j < borderWidth or (octaveLayerIdx < 1) or (octaveLayerIdx > nOctaveLayers):
            return None
    
    if iter >= iterations:
        return None

    j += int(np.round(update_y))
    i += int(np.round(update_x))
    octaveLayerIdx += int(np.round(update_s))
    newResponse = RegionCube[1, 1, 1] + 0.5 * np.dot(gradient, update_list)
    
    # Delete the low response point
    if np.abs(newResponse) * nOctaveLayers < contrastThreshold:
        return None
    
    # Judge whether the point is localize in line
    hessian_xy = hessian_matrix[:2, :2]
    hessian_xy_trace = np.trace(hessian_xy)
    hessian_xy_det = np.linalg.det(hessian_xy)
    if hessian_xy_det < 0 and edgeThreshold * (hessian_xy_trace ** 2) >= (edgeThreshold + 1) ** 2 * hessian_xy_det:
        return None
    
    # Save keypoints
    if UpSampleFlag:
        keypoint = cv2.KeyPoint()
        keypoint.pt = (i * (2 ** octaveIdx), j * (2 ** octaveIdx))
        keypoint.size = sigma * (2 ** (octaveLayerIdx / nOctaveLayers)) * (2 ** (octaveIdx + 1))
        keypoint.octave = octaveIdx + octaveLayerIdx * (2 ** 8) + int(np.round((octaveLayerIdx + 0.5) * 255)) * (2 ** 16)
        keypoint.response = np.abs(newResponse)
        return keypoint, octaveLayerIdx
    else:
        keypoint = cv2.KeyPoint()
        keypoint.pt = (i * (2 ** octaveIdx), j * (2 ** octaveIdx))
        keypoint.size = sigma * (2 ** (octaveLayerIdx / nOctaveLayers)) * (2 ** octaveIdx)
        keypoint.octave = octaveIdx + octaveLayerIdx * (2 ** 8) + int(np.round((octaveLayerIdx + 0.5) * 255)) * (2 ** 16)
        keypoint.response = np.abs(newResponse)
        return keypoint, octaveLayerIdx


def calcKeypointOrientation(gaussianImage, keypoint, upSampleFlag, octaveIdx, num_bins=36, peakRatio=0.8):
    keypointWithOrientaions = []
    h, w = gaussianImage.shape
    # The orientation of the keypoint is calculated on the image layer of the corresponding image octave
    if upSampleFlag:
        curOctaveLayerSigma = keypoint.size / (2 ** (octaveIdx + 1))
    else:
        curOctaveLayerSigma = keypoint.size / (2 ** octaveIdx)
    
    x, y = np.array(keypoint.pt)
    radius = int(np.round(3 * 1.5 * curOctaveLayerSigma))
    gaussianWeightCoefficient = -1 / (2 * (1.5 * curOctaveLayerSigma) ** 2)
    histogram = np.zeros(num_bins)
    smoothHistogram = np.zeros(num_bins)

    for j in range(-radius, radius + 1):
        regionY = int(np.round(y / (2 ** octaveIdx))) + j
        if regionY <= 0 or regionY >= h - 1:
            continue
        for i in range(-radius, radius + 1):
            regionX = int(np.round(x / (2 ** octaveIdx))) + i
            if regionX <= 0 or regionX >= w - 1:
                continue
            # Calculate the gradient magnitude and orientaion
            dx = gaussianImage[regionY, regionX + 1] - gaussianImage[regionY, regionX - 1]
            dy = gaussianImage[regionY + 1, regionX] - gaussianImage[regionY - 1, regionX]
            gradientMagnitude = np.sqrt(dx * dx + dy * dy)
            gradientOrientation = np.rad2deg(np.arctan2(dy, dx))
            gradientOrientationIdx = int(np.round(gradientOrientation * (num_bins / 360.)))
            gaussianWeight = np.exp(gaussianWeightCoefficient * (i ** 2 + j **2))
            histogram[gradientOrientationIdx % num_bins] += gradientMagnitude * gaussianWeight
    
    for i in range(num_bins):
        smoothHistogram[i] = 0.0625 * (histogram[i - 2] + histogram[(i + 2) % num_bins]) + \
                            0.0625 * (4 * (histogram[i - 1] + histogram[(i + 1) % num_bins])) + \
                            0.0625 * (6 * histogram[i])
    maxOrientaionMagnitude = np.max(smoothHistogram)
    
    for peakValueIdx in range(num_bins):
        peakValue = smoothHistogram[peakValueIdx]
        formerPeakValue = smoothHistogram[(peakValueIdx - 1) % num_bins]
        latterPeakValue = smoothHistogram[(peakValueIdx + 1) % num_bins]
        if peakValue < formerPeakValue or peakValue < latterPeakValue:
            continue
        if peakValue <= peakRatio * maxOrientaionMagnitude:
            continue
        # Find the exact peak index using quadratic function interpolate method
        InterpolatePeakIdx = peakValueIdx + ((formerPeakValue - latterPeakValue) / (2 * (formerPeakValue + latterPeakValue - 2 * peakValue)))
        InterpolatePeakIdx = InterpolatePeakIdx % num_bins
        orientation = 360. - InterpolatePeakIdx * 360. / num_bins
        if np.abs(orientation - 360.) < 1e-7:
            orientation = 0
        newKeypoint = cv2.KeyPoint()
        newKeypoint.pt = keypoint.pt
        newKeypoint.size = keypoint.size
        newKeypoint.octave = keypoint.octave
        newKeypoint.response = keypoint.response
        newKeypoint.angle = orientation
        keypointWithOrientaions.append(newKeypoint)
    return keypointWithOrientaions


def calcExtremaPoint(gaussianPyramid, DoGPyramid, sigma, upSampleFlag, borderWidth, nOctaveLayers, contrastThreshold, edgeThreshold, iterations):
    keypoints = []
    threshold = np.floor(0.5 * (contrastThreshold / nOctaveLayers) * 255)
                         
    for octaveIdx, DoGImagesInOctave in enumerate(DoGPyramid):

        for octaveLayerIdx, (firstImage, secondImage, thirdImage) in enumerate(zip(DoGImagesInOctave, DoGImagesInOctave[1:], DoGImagesInOctave[2:])):
            h, w = firstImage.shape
            # Check discrete extrema point in 3 * 3 region
            for j in range(borderWidth, h - borderWidth):
                for i in range(borderWidth, w - borderWidth):
                    firstRegion = firstImage[j - 1 : j + 2, i - 1 : i + 2]
                    secondRegion = secondImage[j - 1 : j + 2, i - 1 : i + 2]
                    thirdRegion = thirdImage[j - 1 : j + 2, i - 1 : i + 2]
                    detectFlag = judgeExtremaPointInDiscreteImage(firstRegion, secondRegion, thirdRegion, threshold)
                    if not detectFlag:
                        continue

                    # The local surface is fitted to calculate sub-pixel point
                    detectResult = calcPointViaQuadraticFit(j, i, octaveLayerIdx, firstImage, secondImage, thirdImage, sigma, octaveIdx, nOctaveLayers, contrastThreshold, edgeThreshold, iterations, upSampleFlag, borderWidth)
                    if detectResult is None:
                        continue
                    keypoint, updateOctaveLayerIdx = detectResult

                    # The orientation of keypoint is calculated on the corresponding image layer
                    gaussianImage = gaussianPyramid[octaveIdx, updateOctaveLayerIdx]
                    keypointWithOrientations = calcKeypointOrientation(gaussianImage, keypoint, upSampleFlag, octaveIdx, num_bins=36, peakRatio=0.8)
                    for keypointWithOrientation in keypointWithOrientations:
                        keypoints.append(keypointWithOrientation)
    return keypoints


def compareKeypoints(keypoint1, keypoint2):
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id


def removeDuplicateKeypoints(keypoints):
    if len(keypoints) < 4:
        return keypoints
    keypoints.sort(key=functools.cmp_to_key(compareKeypoints))
    uniqueKeypoints = [keypoints[0]]

    for curKeypoint in keypoints[1:]:
        formerKeypoint = uniqueKeypoints[-1]
        if (formerKeypoint.pt[0] != curKeypoint.pt[0]) or \
           (formerKeypoint.pt[1] != curKeypoint.pt[1]) or \
           (formerKeypoint.size != curKeypoint.size) or \
           (formerKeypoint.angle != curKeypoint.angle):
            uniqueKeypoints.append(curKeypoint)
    return uniqueKeypoints


def scaleKeypointsToImageSize(keypoints):
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size = keypoint.size * 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
    return keypoints


def nonMaximumSuppression(keypoints, pixelNmsThreshold):
    retainKeypoints = []
    keypoints = sorted(keypoints, key=lambda x : x.response, reverse=True)
    while len(keypoints) > 0:
        curKeypoint = keypoints[0]
        retainKeypoints.append(curKeypoint)
        keypoints = [kpt for kpt in keypoints if np.sqrt((kpt.pt[0] - curKeypoint.pt[0]) ** 2 + (kpt.pt[1] - curKeypoint.pt[1]) ** 2) > pixelNmsThreshold]
    return retainKeypoints


def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / (1 << octave) if octave >= 0 else (1 << -octave)
    return octave, layer, scale


def calcDescriptors(gaussian_pyramid, keypoints, num_bins=8, descr_width=4, scale_factor=3, descr_max_value=0.2):
    descriptors = []
    for keypoint in keypoints:
        octave_idx, layer_idx, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_pyramid[octave_idx + 1, layer_idx]
        h, w = gaussian_image.shape
        # Rescale keypoint to current octave size
        point = np.round(scale * np.array(keypoint.pt)).astype('int')

        bins_per_degree = num_bins / 360.
        angle = 360 - keypoint.angle
        cos_theta = np.cos(np.deg2rad(angle))
        sin_theta = np.sin(np.deg2rad(angle))

        gaussian_coff =  -0.5 / ((0.5 * descr_width) ** 2)
        # 3 * sigma
        hist_width = scale_factor * 0.5 * scale * keypoint.size
        radius = int(hist_width * np.sqrt(2) * (descr_width + 1) * 0.5)
        radius = int(min(radius, np.sqrt(h ** 2 + w ** 2)))
        
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        hist = np.zeros((descr_width + 2, descr_width + 2, num_bins))
        
        for row in range(-radius, radius + 1):
            for col in range(-radius, radius + 1):
                row_rot = col * sin_theta + row * cos_theta
                col_rot = col * cos_theta - row * sin_theta
                row_bin = (row_rot / hist_width) + 0.5 * descr_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * descr_width - 0.5
                if row_bin <= -1 or row_bin >= descr_width or col_bin <= -1 or col_bin >= descr_width:
                    continue
                win_row = int(np.round(point[1] + row))
                win_col = int(np.round(point[0] + col))
                if win_row <= 0 or win_row >= h - 1 or win_col <= 0 or win_col >= w - 1:
                    continue
                dx = gaussian_image[win_row, win_col + 1] - gaussian_image[win_row, win_col - 1]
                dy = gaussian_image[win_row + 1, win_col] - gaussian_image[win_row - 1, win_col]
                grad_magnitude = np.sqrt(dx * dx + dy * dy)
                grad_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                gaussian_weight = np.exp(gaussian_coff * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))

                magnitude_list.append(grad_magnitude * gaussian_weight)
                row_bin_list.append(row_bin)
                col_bin_list.append(col_bin)
                # TODO:
                orientation_bin_list.append((grad_orientation - angle) * bins_per_degree)

        # Trilinear interpolation
        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            hist[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            hist[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            hist[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            hist[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            hist[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            hist[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            hist[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            hist[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = hist[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * descr_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')


def SIFT(image, sigma, upSampleFlag, borderWidth, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, pixelNmsThreshold=2.828, iterations=5, plotFlag=False, saveRoot=None):
    if plotFlag:
        plot_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = image.astype('float')
    h, w = image.shape

    if upSampleFlag:
        # Assuming initial sigma value corresponding the origin resolution image
        _init_sigma = 1.
        sigma = 1.6
        sigmaDiff = np.sqrt(max(sigma ** 2 - _init_sigma ** 2, 0.01))
        image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        baseImage = cv2.GaussianBlur(image, (0, 0), sigmaX=sigmaDiff, sigmaY=sigmaDiff)
    else:
        # Assuming initial sigma value corresponding the origin resolution image
        _init_sigma = 0.5
        sigma = 0.8
        sigmaDiff = np.sqrt(max(sigma ** 2 - _init_sigma ** 2, 0.01))
        baseImage = cv2.GaussianBlur(image, (0, 0), sigmaX=sigmaDiff, sigmaY=sigmaDiff)
    base_h, base_w = baseImage.shape

    if plotFlag:
        cv2.imwrite(osp.join(saveRoot, 'origin_image.png'), image)
        cv2.imwrite(osp.join(saveRoot, 'base_image.png'), baseImage)
    logger.debug(f'Origin image shape: {h} * {w}, Base image shape: {base_h} * {base_w}')

    # Create Gaussian sigmas
    gaussianSigmaList = createGaussianSigma(sigma=sigma, nOctaveLayers=3)
    logger.debug(f'sigmas: {gaussianSigmaList}')

    # Create Gaussian pyramid
    nOctaves = int(np.floor(np.log2(min(base_h, base_w)) - 2))
    logger.debug(f'Num of octaves: {nOctaves}')
    gaussianPyramid = createGaussianPyramid(baseImage, nOctaves, gaussianSigmaList)
    logger.debug(f'Gaussian pyramid shape: {gaussianPyramid.shape}')

    # Calculate DoG
    DoGPyramid = calcDoGPyramid(gaussianPyramid)
    logger.debug(f'DoG Pyramid shape: {DoGPyramid.shape}')

    # Calculate the extrema value
    keypoints = calcExtremaPoint(gaussianPyramid, DoGPyramid, sigma, upSampleFlag, borderWidth, nOctaveLayers, contrastThreshold, edgeThreshold, iterations)
    logger.debug(f'[Detect] The number of keypoints: {len(keypoints)}')
    # Remove duplicate keypoint and NMS
    keypoints = removeDuplicateKeypoints(keypoints)
    logger.debug(f'[Remove duplicate] The number of keypoints: {len(keypoints)}')
    keypoints = nonMaximumSuppression(keypoints, pixelNmsThreshold=pixelNmsThreshold)
    logger.debug(f'[NMS] The number of keypoints: {len(keypoints)}')

    # Scale keypoints to origin image resolution
    if upSampleFlag:
        keypoints = scaleKeypointsToImageSize(keypoints)

    # Calculate the descriptors
    descriptors = calcDescriptors(gaussianPyramid, keypoints, num_bins=8, descr_width=4, scale_factor=3, descr_max_value=0.2)
    logger.debug(f'The descriptors shape: {descriptors.shape}')

    # Plot
    if plotFlag:
        for keypoint in keypoints:
            pt = np.array(keypoint.pt)
            x, y = int(np.round(pt[0])), int(np.round(pt[1]))
            cv2.circle(plot_image, (x, y), radius=2, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(osp.join(saveRoot, 'nms_False.png'), plot_image)

    return keypoints, descriptors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -------------------------------- PATHS -------------------------------- #
    parser.add_argument('--imPath', default='test_data/keypoint_occupy_tiny/i/i_books/3.png')
    # parser.add_argument('--imPath', default='test_data/feature_detect_data/0026.jpeg')
    parser.add_argument('--plotSaveRoot', default='plots/sift_plots/explore_sift_v2')
    # -------------------------------- CONFIGS -------------------------------- #
    parser.add_argument('--upSampleFlag', default=False)
    parser.add_argument('--borderWidth', default=5)
    parser.add_argument('--nfeatures', default=-1)
    parser.add_argument('--nOctaveLayers', default=3)
    parser.add_argument('--contrastThreshold', default=0.04)
    parser.add_argument('--edgeThreshold', default=10)
    parser.add_argument('--pixelNmsThreshold', default=7.07, choices=[2.828, 7.07])
    parser.add_argument('--iterations', default=5)
    # -------------------------------- DEBUG CONFIGS -------------------------------- #
    parser.add_argument('--plotFlag', default=True)
    args = parser.parse_args()

    image = cv2.imread(args.imPath, cv2.IMREAD_GRAYSCALE)
    keypoints, descripotors = SIFT(image, sigma=0, upSampleFlag=args.upSampleFlag, 
                                   borderWidth=args.borderWidth, nfeatures=args.nfeatures, 
                                   nOctaveLayers=args.nOctaveLayers, contrastThreshold=args.contrastThreshold, 
                                   edgeThreshold=args.edgeThreshold, pixelNmsThreshold=args.pixelNmsThreshold, 
                                   plotFlag=args.plotFlag, saveRoot=args.plotSaveRoot)