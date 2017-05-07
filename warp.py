import cv2
import numpy as np

SRC=np.float32(
    [[739, 492],
     [980, 662],
     [298, 662],
     [534, 492]])

DST=np.float32(
    [[980, 0],
     [980, 662],
     [298, 662],
     [298, 0]])

def warp(img, src=SRC, dst=DST):
   img_size = (img.shape[1], img.shape[0])
   M = cv2.getPerspectiveTransform(src, dst)

   m_inverse = cv2.getPerspectiveTransform(dst, src)
   warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
   return warped, m_inverse

if __name__ == '__main__':
    from calibration import calibrate, cal_undistort

    # Test: Undistort and warp straight lines image
    # and plot source and destination points
    original = cv2.imread('test_images/straight_lines1.jpg')
    warped, m_inv = warp(original)
    cv2.imwrite('output_images/warped_output.jpg', warped)

    objpoints, imgpoints = calibrate()
    undistorted = cal_undistort(original, objpoints, imgpoints)

    warped_img, m_inv = warp(undistorted)

    color = [0, 0, 255]
    thickness = 5
    cv2.line(undistorted, (739, 492), (980, 662), color, thickness)
    cv2.line(undistorted, (980, 662), (298, 662), color, thickness)
    cv2.line(undistorted, (298, 662), (534, 492), color, thickness)
    cv2.line(undistorted, (534, 492), (739, 492), color, thickness)

    cv2.line(warped_img, (980, 0), (980, 662), color, thickness)
    cv2.line(warped_img, (980, 662), (298, 662), color, thickness)
    cv2.line(warped_img, (298, 662), (298, 0), color, thickness)
    cv2.line(warped_img, (298, 0), (980, 0), color, thickness)

    cv2.imwrite('output_images/straight_lines_source.jpg', undistorted)
    cv2.imwrite('output_images/straight_lines_destination.jpg', warped_img)
