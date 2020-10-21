import cv2
import numpy as np
import os

w = 11
h = 8
objp = np.zeros((w*h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
target_dir = 'Q1_Image'


def corner_detection():
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
    for img_path in os.listdir(target_dir):
        if not img_path.endswith(('.jpg', '.bmp', '.png', '.JPG')):
            continue
        img_dir = os.path.join(target_dir, img_path)
        print('Processing {}'.format(img_dir))
        img = cv2.imread(img_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # 亞像素精確定位角點位置
        cv2.drawChessboardCorners(img, (11, 8), corners2, ret)  # 圖上繪製檢測到的角點
        print('-' * 50)
        write_dir = os.path.join(target_dir, 'detect')
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        write_path = os.path.join(write_dir, img_path)
        cv2.imwrite(write_path, img)


def intrinsic():
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
    for img_path in os.listdir(target_dir):
        if not img_path.endswith(('.jpg', '.bmp', '.png', '.JPG')):
            continue
        img_dir = os.path.join(target_dir, img_path)
        print('Processing {}'.format(img_dir))
        img = cv2.imread(img_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

        ret, intrinsic_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                           gray.shape[::-1], None, None)
        # rvecs:rotation vector, tvecs:translation vector
    print('* intrinsic_matrix:')
    print(intrinsic_matrix)
    print('-' * 50)


def distortion():
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
    for img_path in os.listdir(target_dir):
        if not img_path.endswith(('.jpg', '.bmp', '.png', '.JPG')):
            continue
        img_dir = os.path.join(target_dir, img_path)
        print('Processing {}'.format(img_dir))
        img = cv2.imread(img_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

        ret, intrinsic_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                           gray.shape[::-1], None, None)
        # rvecs:rotation vector, tvecs:translation vector
    print('* distortion_coefficients:')
    print(distortion_coefficients)
    print('-' * 50)


def extrinsic(img_num: str):
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点

    img_dir = os.path.join(target_dir, (img_num + '.bmp'))
    print('Processing {}'.format(img_dir))
    img = cv2.imread(img_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

    ret, intrinsic_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                           gray.shape[::-1], None, None)
    # rvecs:rotation vector, tvecs:translation vector
    rvecs_R, _ = cv2.Rodrigues(np.array(rvecs))
    extrinsic_matrix = np.concatenate((rvecs_R, np.array(tvecs[0])), axis=1)
    print('* extrinsic_matrix:')
    print(extrinsic_matrix)
    print('-' * 50)