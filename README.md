# **Advanced Lane Finding Project**
Ricardo Solano

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.jpg "Undistorted"
[image2]: ./output_images/undistort_test2.jpg "Road Transformed"
[image3]: ./output_images/binary_output.jpg "Binary Example"
[image4]: ./output_images/straight_lines_source.jpg "Warp Example"
[image5]: ./output_images/straight_lines_destination.jpg "Warp Example"
[image6]: ./output_images/color_fit.png "Fit Visual"
[image7]: ./output_images/histogram.png "Histogram"
[image8]: ./output_images/example_output.jpg "Output"
[image9]: ./output_images/thres_warped_output.png "Thresholded and Perspective Transformed"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate()` and `cal_undistort()` functions (lines 9 through 55) of the `calibration.py` module.  

In `calibrate()` I'm using the code from the writeup template with a modification to serialize/deserialize the chessboard corner computation using the pickle module. This helped speed up code testing while implementing the pipeline.

Basically the code iterates over the calibration images provided, converting each one of them to grayscale and invoking the opencv `findChessboardCorners()` function in order to compute the corner points, which are then saved to a binary file and returned to the caller.

The `cal_undistort()` convenience function takes an image and calculates the distortion coefficients it via the opencv `calibrateCamera()` function using the precomputed corner data points from the previous step. Then the resulting coefficients are used by the opencv `undistort()` function to correct the image. This is what my distortion-corrected chessboard test image looks like:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The following is a distortion-corrected version of the test2.jpg example image provided in this assignment:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for image binary thresholding is contained in the `thresholding.py` module, lines 4 to 95. In order to generate a binary image I used Sobel, gradient magnitude, gradient direction and S-channel thresholds.

The `bin_thres_img()` function combines all four thresholding methods and outputs the final thresholded binary image.

Here is an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My perspective transform code is contained in the `warp()` function in lines 16 to 22 of the `warp.py` module.

The `warp()` function takes an image, source and destination points and returns the warped image. There the opencv `getPerspectiveTransform()` function generates a transform matrix using the source/destination points. This matrix is then passed to the `warpPerspective()` function which takes care of producing a warped image. 

I manually picked the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 690, 450      | 1100, 0       | 
| 1100, 720     | 1100, 720     |
| 200, 720      | 200, 720      |
| 590, 450      | 200, 0        |

I verified that my perspective transform was working as expected by drawing the source and destination points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Original Image             |  Undistorted & Warped Image
:-------------------------:|:-------------------------:
![alt text][image4]        | ![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane line detection step is performed on a *calibrated, binary thresholded and perspective transformed (bird's-eye view) image* generated using the functions in the previous steps.
![alt text][image9]

The code for this step lives in the `pipeline.py` module, `lane_fit()` and `faster_lane_fit()` functions in lines 4-138.

Initial lane line detection is done using the *histogram* method: pixel values are added up along the columns in the bottom half of the image. This results in two identifiable peaks which can be used as the base of the lane lines.
![alt text][image7]

I then use the *sliding window* approach to identify the x,y values for the left and right lane lines and fit a second order polynomial to each set of line pixel positions.

![alt text][image6]

Note: I had trouble getting the lines to display from the binary warped image after plotting.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is in the `calc_curvature()` and `calc_offset()` functions, lines 188 through 239 of the `pipeline.py` module.

The curvature is calculated by converting the lane line pixel values detected to world space and fitting a second order polynomial on the resulting left and right values. Then I use the equation for the radius of the curvature `R=(1+(2Ay+B)*2)*1.5/2A` to get the curvature values for each lane line. The pixel to meter conversion values used are 30/720 meters per pixel in y dimension and 3.7/700 meters per pixel in x dimension.

For the vehicle offset lane calculation I assume the camera is at the center of the image. The lane center is calculated from the detected left and right lane points closest to the vehicle (highest y value). The offset is the difference between the camera position and the lane center. The value is also converted to meters using the same scale as in the curvature step.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 140 through 186 in my code in `pipeline.py` in the function `draw_detection()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline implementation reuses most code from the classroom lessons and quizzes. The code has been refactored into modules and functions for better readability and reusability and to facilitate the testing of each functionality. Once my pipeline was nearly complete, I had some issues properly detecting lanes from the video at approximately 39 seconds in. I solved this by averaging out the lane fits for the previous 25 frames. Implementing the Line class as suggested in the classroom allowed me to keep a history of lane detection data points. Still additional thresholding would have probably helped better detect the lanes under certain lighting conditions. Possibly L channel thresholding would have improved detection in frames with excessive brightness. Other situations where the pipeline would like fail include the presence of objects in the lane, such as other vehicles, or even lines painted over the original lanes during road works. In order to make the pipeline more robust, additional object detection would need to be baked into the algorithm to allow differentiating lane lines from other elements.
