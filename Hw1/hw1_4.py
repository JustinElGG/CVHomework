import numpy as np
import cv2
import os

target_dir = 'Q4_Image'
img1_path = os.path.join(target_dir, 'Aerial1.jpg')
img1_savepath = os.path.join(target_dir, 'FeatureAerial1.jpg')
img1 = cv2.imread(img1_path, 0)
img2_path = os.path.join(target_dir, 'Aerial2.jpg')
img2_savepath = os.path.join(target_dir, 'FeatureAerial2.jpg')
img2 = cv2.imread(img2_path, 0)
img3_savepath = os.path.join(target_dir, 'FeatureMatch.jpg')


def SIFT(condition):
    sift = cv2.xfeatures2d.SIFT_create()  # sift object
    keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()  # matcher
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    '''  # alternative matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor1, descriptor2, k=2)
    '''

    if condition == 'keypoint':
        good = [m for m, n in matches if m.distance < 0.18 * n.distance]
        kp1_matched = ([ keypoints1[m.queryIdx] for m in good[:6] ])
        kp2_matched = ([ keypoints2[m.trainIdx] for m in good[:6] ])
        img1_show = cv2.drawKeypoints(img1, kp1_matched, None, color=(255, 0, 255), flags=0)
        img2_show = cv2.drawKeypoints(img2, kp2_matched, None, color=(255, 0, 255), flags=0)
        cv2.imwrite(img1_savepath, img1_show)
        cv2.imwrite(img2_savepath, img2_show)
        numpy_horizontal = np.hstack((img1_show, img2_show))
        cv2.imshow('key', numpy_horizontal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif condition == 'match':
        good = [[m] for m, n in matches if m.distance < 0.18 * n.distance]
        img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good[:6], None, flags=2)
        cv2.imshow('key', img3)
        cv2.imwrite(img3_savepath, img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# SIFT('keypoint')
# SIFT('match')
