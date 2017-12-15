# -*- coding: utf-8 -*-
from multiprocessing import Queue

#Location of the input test images
locationImageFiles = "./testImages"
locationOfInputVideos = "./InputVideos"
locationOfOutputVideos = "./OutputVideos"

# For Camera Calibration
DEBUG = False
CHESSBOARD_COLUMNS = 9
CHESSBOARD_ROWS = 6
RELATIVE_PATH_TO_CALIBRATION_IMAGES="./Camera_Calibration_InputImages/"
RELATIVE_PATH_TO_OUTPUT_CALIBRATION_FOLDER="./Camera_Calibration_OutputImages/"
NOMINAL_HEIGHT_OF_CALIBRATION_IMAGES=720
NOMINAL_WIDTH_OF_CALIBRATION_IMAGES=1280




#Location for storing output processed images from the lane detection routines
locationProcessedImages = "./Results"
imageOffsetFromTop = int(0.45*720)

#Performance Variables for image processing
normalize = False
Clip = True
Blur = False
Contrast = True
HOG = False
Histogram = False
colorSpace = 'RGB'
CannyLowerThreshold = 150.0
CannyUpperThreshold = 500.0
Histogram_Channel = []
ChannelstoNormalize = []
nThreads = 2
channel = Queue(maxsize=1)
ClipOffsetXBottomLeft = 0.2
ClipOffsetXTopLeft = 0.45
ClipOffsetXTopRight = 0.45
ClipOffsetXBottomRight = 0.15
ClipOffsetYTop = 0.55
ClipOffsetYBottom = 0.1
GlobalNormalization = False
pixelRanges = []
nProcesses = 7
doCanny = False
doBlur = False
BlurKernelSize = 3
houghThreshold = 20.0
MinLineLength = 150.0
MaxLineGap = 500.0
doRelativeIsolation = False
directionalGradient = True
DirectionalGradientKernelSize = 7
DirectionalGradientLowerThreshold = 0.785398
DirectionalGradientUpperThreshold = 1.57
RelativePathtoCameraMatrix = "./Camera_Calibration_OutputImages/"
doHOG = False
#HOG Defaults
nHOGOrientations = 2 
nHOGPixelsPerCellX = 50
nHOGPixelsPerCellY = 72
nHOGCellsPerBlockX = 1
nHOGCellsPerBlockY = 1
camera_matrix = []
dist_coefs = []

#Histogram Based Lane Marking
doHistogramBasedLaneMarking = True
nHistogramWindows = 9
