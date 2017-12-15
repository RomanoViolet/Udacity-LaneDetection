## Lane Detection Using Computer Vision

### About
This project implements Lane Detection and Advanced Lane Detection projects as required by the Udacity's [Self Driving Car Nano-Degree program.](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

Lane Detection is done using the [Python](https://www.python.org/) implementation of the [Open Source Computer Vision Library (OpenCV)](https://opencv.org/).

### How it Looks
A sample frame from one of the videos overlaid with markers is ![shown](https://github.com/RomanoViolet/Udacity-LaneDetection/blob/master/Results/vlcsnap-2017-12-16-00h00m30s932.png)

### Structure of the Project
The implementation is split into four files:
- LaneDetection.py: Entry point for the implentation. Read caveat first.
- LaneDetectionUtils.py: Image processing helper functions used for lane detection
- calibrateCamera.py: Routine for compensating for camera distortion before processing images and video.
- configuration.py: Configures the steps required to execute the lane detection pipeline.

Necessary input data is provided in:
- Camera_Calibration_InputImages: Set of images taken from the camera which is used in the vehicle. These images are distorted (e.g., have spherical distortion component). A standard checker-board pattern is used for computing camera calibration coefficients.
- testImages: Captured from the assignement videos provided as part of the assignments. The images are not annotated with lanes markers.
- InputVideos: A set of three evaluation videos, with increasing levels of difficultly.

Folders which may be used to hold the result of the lane-detection implementation are appropriately named.

### Running the Lane Detection Application
Execute `run.sh` included in the project. Tested on *nix, kernel 4.x, Python 3.5.2

### Caveat
Multiprocessing library has been used to speed up image and video processing. However, it was noticed that calling `CalibrateCamera()` routine from within the `LaneDetection.py` caused threaded workers to hang on a futex. See comments in lines 348-351 in LaneDetection.py. Corrections welcome.

### Exploring Options
- It is possible to configure the sequence of steps executed for detecting lanes by setting or modifying options from _configuration.py_. Normally, no other changes are required. The changes can be made at runtime -- simply modify your choice of option, and save the _configuration.py_. The effect of all changes are visible after the current pool of threads returns.

- In order to evaluate the effect of the choice made, the LaneDetection continuously cycles through the images in the testImages folder, in case `doLaneDetection()` option is uncommented out.

- Set the number of threads allowed by tweaking the option `nProcesses` from the _configuration.py_ file.

### Pending Improvements
- The current pipeline works well for "project_video", and is acceptable for "challenge_video", but is poor for "harder_challenge_video". Suggestions welcome.
- The logic can be reorganized for readability.

### Credits
- Udacity: Lecturers, and mentors;
- Internet: for examples and samples.

### Disclaimer
Some of the ideas are borrowed and adapted from other people's work.
