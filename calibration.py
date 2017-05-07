import cv2
import glob
import numpy as np
import pickle

CALIBRATION_IMAGES = 'camera_cal/calibration*.jpg'
CALIBRATION_DATA = 'calibration.p'

def calibrate(calibration_images=CALIBRATION_IMAGES):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    try:
        with open(CALIBRATION_DATA, "rb") as f:
            print("calibration file found")
            objpoints = pickle.load(f)
            imgpoints = pickle.load(f)
            return objpoints, imgpoints
    except:
        print("no calibration file, calibrating...")
        pass

    # Make a list of calibration images
    images = glob.glob(calibration_images)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    with open(CALIBRATION_DATA, "wb") as f:
        print("saving calibration data...")
        pickle.dump(objpoints, f)
        pickle.dump(imgpoints, f)

    return objpoints, imgpoints

def cal_undistort(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, None)
    return undist

if __name__ == '__main__':
    cal_test = cv2.imread('camera_cal/calibration1.jpg')
    objpoints, imgpoints = calibrate()
    undistorted = cal_undistort(cal_test, objpoints, imgpoints)
    cv2.imwrite('output_images/undistort_output.jpg', undistorted)
