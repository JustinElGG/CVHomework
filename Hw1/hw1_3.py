import numpy as np
import sys
import cv2
import os
from matplotlib import pyplot as plt

target_dir = 'Q3_Image'
np.set_printoptions(threshold=sys.maxsize)


def disparity_map():
    imgL_path = os.path.join(target_dir, 'imL.png')
    imgR_path = os.path.join(target_dir, 'imR.png')
    imgL = cv2.imread(imgL_path, 0)
    imgR = cv2.imread(imgR_path, 0)
    imgL = cv2.resize(imgL, (255, 191))
    imgR = cv2.resize(imgR, (255, 191))
    stereo = cv2.StereoBM_create(16, 15)
    disparity = stereo.compute(imgL, imgR)
    disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity = disparity.astype(np.uint8)
    save_path = os.path.join(target_dir, 'result.png')
    cv2.imshow('Disparity Map', disparity)
    cv2.imwrite(save_path, disparity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
