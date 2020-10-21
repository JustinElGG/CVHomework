import cv2
import numpy as np
import os

w = 11
h = 8
objp = np.zeros((w*h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
target_dir = 'Q2_Image'

src = np.float32([[3, 3, -3], [1, 1, 0], [3, 5, 0], [5, 1, 0]]).reshape(-1, 3)


def get_coff():
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
    for img_path in os.listdir(target_dir):
        if not img_path.endswith(('.jpg', '.bmp', '.png', '.JPG')):
            continue
        img_dir = os.path.join(target_dir, img_path)
        img = cv2.imread(img_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # 亞像素精確定位角點位置
        cv2.drawChessboardCorners(img, (11, 8), corners, ret)  # 圖上繪製檢測到的角點
        ret, intrinsic_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                           gray.shape[::-1], None, None)
        # rvecs:rotation vector, tvecs:translation vector
    return intrinsic_matrix, distortion_coefficients


active = True
def inputHandler(value):
    global active
    if value == 'exit':
        active = False


def draw_tri():
    global active
    active = True
    intrinsic_matrix, distortion_coefficients = get_coff()
    img_counter = 0
    while active:
        if img_counter < 5:
            img_counter += 1
        else:
            img_counter = 1
        objpoints = []  # 在世界坐标系中的三维点
        imgpoints = []  # 在图像平面的二维点
        img_dir = os.path.join(target_dir, (str(img_counter) + '.bmp'))
        print('Processing {}'.format(img_dir))
        img = cv2.imread(img_dir)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        ret, _, _, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        imgpts, _ = cv2.projectPoints(src, np.array(rvecs), np.array(tvecs), intrinsic_matrix, distortion_coefficients)
        imgpts = imgpts.astype(int)
        frame = cv2.imread(img_dir)
        cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 2)
        cv2.line(frame, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 2)
        cv2.line(frame, tuple(imgpts[2].ravel()), tuple(imgpts[0].ravel()), (0, 0, 255), 2)

        cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 2)
        cv2.line(frame, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 2)
        cv2.line(frame, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 2)

        frame = cv2.resize(frame, (768, 768))
        cv2.imshow('h', frame)
        cv2.waitKey(500)
        print('-' * 50)
    cv2.destroyAllWindows()
