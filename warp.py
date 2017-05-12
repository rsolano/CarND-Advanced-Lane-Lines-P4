import cv2
import numpy as np

# Source points
src_tr = [690, 450]
src_br = [1100, 720]
src_bl = [200, 720]
src_tl = [590, 450]

# Destination points
dst_tr = [1100, 0]
dst_br = [1100, 720]
dst_bl = [200, 720]
dst_tl = [200, 0]

SRC=np.float32(
    [src_tr,
     src_br,
     src_bl,
     src_tl])

DST=np.float32(
    [dst_tr,
     dst_br,
     dst_bl,
     dst_tl])

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
    cv2.line(undistorted, tuple(src_tr), tuple(src_br), color, thickness)
    cv2.line(undistorted, tuple(src_br), tuple(src_bl), color, thickness)
    cv2.line(undistorted, tuple(src_bl), tuple(src_tl), color, thickness)
    cv2.line(undistorted, tuple(src_tl), tuple(src_tr), color, thickness)
    cv2.line(warped_img, tuple(dst_tr), tuple(dst_br), color, thickness)
    cv2.line(warped_img, tuple(dst_br), tuple(dst_bl), color, thickness)
    cv2.line(warped_img, tuple(dst_bl), tuple(dst_tl), color, thickness)
    cv2.line(warped_img, tuple(dst_tl), tuple(dst_tr), color, thickness)

    cv2.imwrite('output_images/straight_lines_source.jpg', undistorted)
    cv2.imwrite('output_images/straight_lines_destination.jpg', warped_img)
