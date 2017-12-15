## Lane Detection Using Computer Vision

### About
This project implements Lane Detection and Advanced Lane Detection projects as required by the Udacity's [Self Driving Car Nano-Degree program.](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

Lane Detection is done using the [Python](https://www.python.org/) implementation of the [Open Source Computer Vision Library (OpenCV)](https://opencv.org/).

### How it Looks
A sample frame from one of the videos overlaid with markers is shown below:
![Image](src)

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

You can use the [editor on GitHub](https://github.com/RomanoViolet/Udacity-LaneDetection/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/RomanoViolet/Udacity-LaneDetection/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
