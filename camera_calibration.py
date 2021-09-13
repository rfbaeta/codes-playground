import cv2
import numpy as np
import os
import glob

#https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615
data = "./data/camera_calibration_syt"

def calibrate_chessboard(dir_path, image_format, square_size, width, height):
    cont = 0
    '''Calibrate a camera using chessboard images.'''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(os.path.join(dir_path, f"*.{image_format}"))

    # Iterate through all images
    for fname in images:
        print(f"Reading {fname}")
        img = cv2.imread(str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"./data/camera_calibration/gray_{cont}.jpg", gray)
        cont+=1
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6),  None)
        print(ret)
        print(corners)
        # If found, add object points, image points (after refining them)
        if ret:
            fnl = cv2.drawChessboardCorners(img, (7, 7), corners, ret)
            cv2.imwrite(f"./data/camera_calibration/corners_{cont}.jpg", fnl)
            #objpoints.append(objp)

        #    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
         #   imgpoints.append(corners2)

    # Calibrate camera
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return []
    #return [ret, mtx, dist, rvecs, tvecs]

parameters = calibrate_chessboard(data, "jpg", 4, 9, 6)