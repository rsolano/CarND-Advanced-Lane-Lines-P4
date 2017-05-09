# **Advanced Lane Finding Project**

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

[image6]: ./examples/color_fit_lines.jpg "Fit Visual"
[image7]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate()` and `cal_undistort()` functions (lines 9 through 55) of the `calibration.py` module.  

In `calibrate()` I'm using the code from the writeup template with a modification to serialize/deserialize the chessboard corner computation using the pickle module. This helped speed up code testing while implementing the pipeline.

Basically the code iterates over the calibration images provided, converting each one of them to grayscale and invoking the opencv `findChessboardCorners()` function in order to compute the corner points, which are then saved to a binary file and returned to the caller.

The `cal_undistort()` convenience function takes an image and calculates the distortion coefficients it via the opencv `calibrateCamera()` function using the precomputed corner data points from the previous step. Then the resulting coefficients are used by the opencv `undistort()` function to correct the image. This is what my distortion-corrected chessboard test image looks like:

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My pipeline implementation reuses most code from the classroom lessons and quizzes. The code has been refactored into modules for better readability and facilitate the testing of each step. 

The code for image binary thresholding is contained in the `thresholding.py` module, lines 4 to 95. In order to generate a binary image I used Sobel, gradient magnitude, gradient direction and S-channel thresholds. Here is an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My perspective transform code is contained in the `warp()` in lines 16 to 22 of the warp.py module.

The `warp()` function takes an image, source and destination points and returns the warped image as well as the inverse matrix.

I manually picked the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 739, 492      | 980, 0        | 
| 980, 662      | 980, 662      |
| 298, 662      | 298, 662      |
| 534, 492      | 298, 0        |

I verified that my perspective transform was working as expected by drawing the source and destination points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4] ![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
