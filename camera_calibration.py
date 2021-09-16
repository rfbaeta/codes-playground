import cv2
import numpy as np
import os
import glob

#https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615
data = "./data/camera_calibration_syt"
def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given path/file.'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


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
        ret, corners = cv2.findChessboardCorners(gray, (11, 7),  criteria)
        # If found, add object points, image points (after refining them)
        if ret:
            fnl = cv2.drawChessboardCorners(img, (11, 7), corners, ret)
            cv2.imwrite(f"./data/camera_calibration/corners_{cont}.jpg", fnl)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 7), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return [ret, mtx, dist, rvecs, tvecs]

parameters = calibrate_chessboard(data, "jpg", 4, 11, 7)
save_coefficients(parameters[1], parameters[2], "calibration_chessboard.yml")