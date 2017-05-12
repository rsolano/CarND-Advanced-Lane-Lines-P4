import numpy as np
import cv2

def lane_fit(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if DEBUG:
        print("\nNormal lane fit")
        print("  Left inds", len(left_lane_inds))
        print("  Right inds", len(right_lane_inds))
        print("  Lefty", lefty.shape)
        print("  Leftx", leftx.shape)
        print("  Righty", righty.shape)
        print("  Rightx", rightx.shape)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Update Line objects
    left_line.detected = True
    left_line.current_fit = left_fit
    left_line.allx = leftx
    left_line.ally = lefty

    right_line.detected = True
    right_line.current_fit = right_fit
    right_line.allx = rightx
    right_line.ally = righty

def faster_lane_fit(binary_warped):
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    if DEBUG:
        print("\nFaster lane fit")
        print("  Left inds", len(left_lane_inds))
        print("  Right inds", len(right_lane_inds))
        print("  Lefty", lefty.shape)
        print("  Leftx", leftx.shape)
        print("  Righty", righty.shape)
        print("  Rightx", rightx.shape)

    # if len(left_lane_inds) < 41000:
    if (leftx.shape[0] < 10000 or rightx.shape[0] < 10000):
        left_line.detected, right_line.detected = False, False
        return False

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Update Line objects
    left_line.current_fit = left_fit
    left_line.allx = leftx
    left_line.ally = lefty

    right_line.current_fit = right_fit
    right_line.allx = rightx
    right_line.ally = righty
    return True

def draw_detection(image, warped, m_inverse):
    undist = cal_undistort(image, objpoints, imgpoints)

    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_line.recent_xfitted.append(left_fitx)
    right_line.recent_xfitted.append(right_fitx)

    avg_left_fitx = sum(left_line.recent_xfitted) / len(left_line.recent_xfitted)
    avg_right_fitx = sum(right_line.recent_xfitted) / len(right_line.recent_xfitted)

    curvature = calc_curvature(ploty)
    vehicle_offset = calc_offset(image)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([avg_left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([avg_right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inverse, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # Write text
    radius_text = "Radius of curvature: {:.2f}m".format(curvature)
    offset_text = "Vehicle offset: {:.2f}m".format(vehicle_offset)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255,255,255)
    line_type = cv2.LINE_AA
    cv2.putText(result, radius_text, (50,50), font, 1, font_color, 2, line_type)
    cv2.putText(result, offset_text, (50,90), font, 1, font_color, 2, line_type)
    cv2.putText(result, "RS", (1200,700), font, 1, font_color, 2, line_type)

    return result

def calc_curvature(ploty):
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit
    leftx = left_line.allx
    lefty = left_line.ally
    rightx = right_line.allx
    righty = right_line.ally

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty) / 2
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    if DEBUG:
        print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curvature = (left_curverad + right_curverad) / 2

    # Now our radius of curvature is in meters
    if DEBUG:
        print(left_curverad, 'm', right_curverad, 'm')
        print("Curvature:", curvature)

    return curvature

def calc_offset(image):
    right_x_predictions = right_line.allx
    left_x_predictions = left_line.allx

    camera_position = image.shape[1]/2
    lane_center = (right_x_predictions[-1] + left_x_predictions[-1])/2
    center_offset_pixels = camera_position - lane_center

    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    result = center_offset_pixels * xm_per_pix

    if DEBUG:
        print("image shape", image.shape)
        print("px offset", center_offset_pixels)
        print("m offset", result)

    return result

def pipeline(image):
    undist = cal_undistort(image, objpoints, imgpoints)
    warped, m_inverse = warp(undist)
    binary_warped = bin_thres_img(warped)

    detected = left_line.detected and right_line.detected
    if not detected:
        lane_fit(binary_warped)
    else:
        faster_lane_fit(binary_warped)

    return draw_detection(image, binary_warped, m_inverse)

if __name__ == '__main__':
    from calibration import calibrate, cal_undistort
    from warp import warp
    from line import Line
    from thresholding import *

    import glob
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    from moviepy.editor import VideoFileClip
    from IPython.display import HTML

    DEBUG = True

    objpoints, imgpoints = calibrate()
    left_line = Line()
    right_line = Line()

    output = 'project_video_out.mp4'
    clip = VideoFileClip('project_video.mp4')
    out_clip = clip.fl_image(pipeline)
    out_clip.write_videofile(output, audio=False)
