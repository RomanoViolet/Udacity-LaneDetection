# -*- coding: utf-8 -*-
from multiprocessing import Queue

#Location of the input test images
locationImageFiles = "./testImages"
locationOfInputVideos = "../InputVideos"
locationOfOutputVideos = "../OutputVideos"

#Location for storing output processed images
locationProcessedImages = "./Results"
imageOffsetFromTop = int(0.45*720)
#Performance Variables
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
RelativePathtoCameraMatrix = "./Calibrations"
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
