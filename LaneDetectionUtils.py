#LaneDetectionUtils.py
import os
import sys
import cv2
import numpy as np
np.seterr(all='raise')
import pickle
import configuration
from skimage.feature import hog
from skimage import exposure
from scipy import ndimage

'''
Read in stored camera calibrations
'''
def getCalibrationCoefficients(RelativePathtoCameraMatrix):
    pathtoCameraCoefficients = os.path.join(RelativePathtoCameraMatrix, "wide_dist_pickle.p")
    dist_pickle = pickle.load( open( pathtoCameraCoefficients, "rb" ) )
    camera_matrix = dist_pickle["mtx"]
    dist_coefs = dist_pickle["dist"]
    
    return camera_matrix,  dist_coefs


def customContrast(img):
    #r = currentImage[:, :, 3]
    #g = currentImage[:, :, 2]
    #b = currentImage[:, :, 1]
    
    #Muliply each pixel by 0.0041857.pixel^2  -0.0646125*pixel + 0.3078798
    #coeff = [0.0041857, -0.0646125,  0.3078798]
    #contrastedImage = coeff[0]*img*img + coeff[1]*img + coeff[2]
    
    Yellow = np.zeros_like(img)
    red_Channel = img[:, :, 0]
    green_Channel = img[:, :, 1]
    blue_Channel = img[:, :, 2]
    Yellow[ (red_Channel > 150) & (green_Channel > 150) & (green_Channel >= 0.65*red_Channel) & (green_Channel < 1.35*red_Channel) & (blue_Channel<0.7*(np.maximum(red_Channel, green_Channel)))] = 1
    
    White = np.zeros_like(img)
    White[(green_Channel >= 175) & (blue_Channel >= 175) & (blue_Channel>=175)] = 1
    
    contrastedImage = np.zeros_like(img)
    
    contrastedImage[ (White==1) | (Yellow==1)] = 255
    
    
    return contrastedImage.astype(np.uint8)
    
    
    
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image



def clipImages(img):

    heightOfImage = img.shape[0]
    widthOfImage = img.shape[1]
    
    #Correct for any oddly shaped images
    if((heightOfImage/720. < 1.1) and (heightOfImage/720. > 0.9)):
        img = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_AREA)
    
    
    originalImage = np.copy(img)
    #We resized the image to 1280x720
    heightOfImage = 720
    widthOfImage = 1280

    #Create filter vertices
    #Set vertices of masking polygon
    #Offset mask horizontally from bottom left of image
    horizontalMaskOffsetatBottomLeft =  configuration.ClipOffsetXBottomLeft

    #Offset mask horizontally from bottom right of image
    horizontalMaskOffsetatBottomRight = configuration.ClipOffsetXBottomRight

    #Offset mask horizontally from top left of image
    horizontalMaskOffsetatTopLeft = configuration.ClipOffsetXTopLeft 

    #Offset mask horizontally from top right of image
    horizontalMaskOffsetatTopRight = configuration.ClipOffsetXTopRight

    #Offset mask from top left of image
    VerticalMaskOffsetatTop = configuration.ClipOffsetYTop

    #Offset mask from top right of image
    VerticalMaskOffsetatBottom = configuration.ClipOffsetYBottom

    #print("[From Clipper] Clipping: Bottom Left X: %f"%(horizontalMaskOffsetatBottomLeft))

    vertices = np.array([
                 [
                         #Bottom left vertex
                         (horizontalMaskOffsetatBottomLeft*widthOfImage, heightOfImage-(VerticalMaskOffsetatBottom*heightOfImage)),
                         
                         #Top left vertex
                         (horizontalMaskOffsetatTopLeft*widthOfImage, (VerticalMaskOffsetatTop*heightOfImage)),
                         
                         #Top Right vertex
                         (widthOfImage - horizontalMaskOffsetatTopRight*widthOfImage, (VerticalMaskOffsetatTop*heightOfImage)),
                         
                         #Bottom right vertex
                         (widthOfImage - horizontalMaskOffsetatBottomRight*widthOfImage, heightOfImage-(VerticalMaskOffsetatBottom*heightOfImage))
                 ]       
            
            ], dtype=np.int32)

    
    clippedImage = region_of_interest(img, vertices)
    
    return originalImage, clippedImage


def normalizeImages(img,  channels,  globalNormalization):
    normalizedImage  = np.copy(img)
    for channel in channels:
        #Local normalization
        ChannelMean = np.mean(np.asarray(img[:,:,channel]).astype(float), axis=(0,1), keepdims=True)
        ChannelStd = np.std(np.asarray(img[:,:,channel]).astype(float), axis=(0,1), keepdims=True)
        #ChannelNormalized = (np.asarray(img[:,:,channel]).astype(float) - ChannelMean) / float(ChannelStd)
        ChannelNormalized = (np.asarray(img[:,:,channel]).astype(float) - 0.*ChannelMean) / float(ChannelStd)
        
        normalizedImage = np.copy(img.astype(np.uint8))
        
        normalizedImage[:,:,channel] = (ChannelNormalized.astype(np.uint8))
    
    if(globalNormalization):
        globalMean = np.mean(np.asarray(normalizedImage).astype(float), axis=(0,1,2), keepdims=True)
        globalStd = np.std(np.asarray(normalizedImage).astype(float), axis=(0,1, 2), keepdims=True)
        normalizedImage = (normalizedImage- 0.*globalMean) / float(globalStd)
    
        
    return np.asarray(normalizedImage.astype(np.uint8))
    
    
def changeColorSpace(targetColorSpace,  img):
    
    if targetColorSpace != 'RGB':
        if targetColorSpace == 'YUV':
            imageWithTargetColorSpace = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        
        elif targetColorSpace == 'HSV':
            imageWithTargetColorSpace = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        elif targetColorSpace == 'LUV':
            imageWithTargetColorSpace = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        
        elif targetColorSpace == 'HLS':
            imageWithTargetColorSpace = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        elif targetColorSpace == 'YCrCb':
            imageWithTargetColorSpace = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            
        elif targetColorSpace == 'CYMK':
            imageWithTargetColorSpace = []
            imageWithTargetColorSpace = np.copy(img)
            imageWithTargetColorSpace[:,:,0] = 0*imageWithTargetColorSpace[:,:,0]
            imageWithTargetColorSpace[:,:,1] = 0*imageWithTargetColorSpace[:,:,0]
            imageWithTargetColorSpace[:,:,2] = 0*imageWithTargetColorSpace[:,:,0]
            imageWithTargetColorSpace = np.dstack((imageWithTargetColorSpace,0*imageWithTargetColorSpace[:,:,0]))
            #http://stackoverflow.com/questions/14088375/how-can-i-convert-rgb-to-cmyk-and-vice-versa-in-python
            cmyk_scale = 100
            #CV arranges channels in B-G-R order
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            
            if (np.all(r==0)) and (np.all(g==0)) and (np.all(b==0)):
                # black
                return 0, 0, 0, cmyk_scale

            # rgb [0,255] -> cmy [0,1]
            c = 1 - r / 255.
            m = 1 - g / 255.
            y = 1 - b / 255.

            # extract out k [0,1]
            min_cmy = 0.01+np.minimum(c, m, y)
            c = (c - min_cmy) / (1 - min_cmy)
            m = (m - min_cmy) / (1 - min_cmy)
            y = (y - min_cmy) / (1 - min_cmy)
            k = min_cmy

            # rescale to the range [0,cmyk_scale]
            imageWithTargetColorSpace[:,:,0] = c*cmyk_scale
            imageWithTargetColorSpace[:,:,1] = m*cmyk_scale
            imageWithTargetColorSpace[:,:,2] = y*cmyk_scale
            imageWithTargetColorSpace[:,:,3] = k*cmyk_scale
            #return c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale
            
            #Drop C channel, as we are operating only with 3 channels
            imageWithTargetColorSpace = imageWithTargetColorSpace[:, :, 1:4]
        
        elif targetColorSpace == 'Gray':
            imageWithTargetColorSpace = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            #Go from 1 channel to 3 channels
            imageWithTargetColorSpace = cv2.cvtColor(imageWithTargetColorSpace, cv2.COLOR_GRAY2RGB)
            
            
            
    else: 
        imageWithTargetColorSpace = np.copy(img)
        
    return imageWithTargetColorSpace


def isolatePixelsPerChannel(img,  pixelRanges):
    pixelRangesPerChannel = []
    for channel in range(2, -1,  -1):
        pixelRangesPerChannel.append([pixelRanges[channel*2+1],  pixelRanges[channel*2+0]])
        
        
    imageWithIsolatedPixels = np.zeros_like(img)
    channel0 = imageWithIsolatedPixels[:, :, 0]
    channel1 = imageWithIsolatedPixels[:, :, 1]
    channel2 = imageWithIsolatedPixels[:, :, 2]
    
    channel0= (img[:, :, 0]>=pixelRangesPerChannel[0][0]) & (img[:, :, 0]<=pixelRangesPerChannel[0][1])
    channel1= (img[:, :, 1]>=pixelRangesPerChannel[1][0]) & (img[:, :, 1]<=pixelRangesPerChannel[1][1])
    channel2= (img[:, :, 2]>=pixelRangesPerChannel[2][0]) & (img[:, :, 2]<=pixelRangesPerChannel[2][1])
    
    imageWithIsolatedPixels[:, :, 0] = (channel0*255).astype(np.uint8)
    imageWithIsolatedPixels[:, :, 1] = (channel1*255).astype(np.uint8)
    imageWithIsolatedPixels[:, :, 2] = (channel2*255).astype(np.uint8)
    
    return imageWithIsolatedPixels


def customIsolatePixel(img,  pixelRanges):
    
    imageWithIsolatedPixels = np.zeros_like(img)
    
    #For channel 0
    localImage_channel0 = img[:, :, 0]
    localImage_channel1 = img[:, :, 1]
    localImage_channel2 = img[:, :, 2]
    
    meanValue_channel0 = np.mean(localImage_channel0)
    meanValue_channel1 = np.mean(localImage_channel1)
    meanValue_channel2 = np.mean(localImage_channel2)
    
    #channel0 = (localImage_channel0[:, :, 0]< 0.25* meanValue_channel0)
    #channel1 = (localImage_channel0[:, :, 1]< 0.25* meanValue_channel1)
    #channel2 = (localImage_channel0[:, :, 2]< 0.25* meanValue_channel2)
    
    channel0 = (img[:, :, 0]< 0.25* meanValue_channel0)
    channel1 = (img[:, :, 1]< 0.25* meanValue_channel1)
    channel2 = (img[:, :, 2]< 0.25* meanValue_channel2)
    
    imageWithIsolatedPixels[:, :, 0] = (channel0*255).astype(np.uint8)
    imageWithIsolatedPixels[:, :, 1] = (channel1*255).astype(np.uint8)
    imageWithIsolatedPixels[:, :, 2] = (channel2*255).astype(np.uint8)
    
    return imageWithIsolatedPixels


def draw_lines(img, lines, color=[255, 255, 0], thickness=3):
    
    if(lines is not None):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


    
def hough_lines(img, threshold, min_line_len, max_line_gap, rho = 1, theta = np.pi/180):
    #lines = []
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    draw_lines(line_img, lines)
    
    return line_img, lines


                
                
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)
    
    
    
def doEdgeDetection(img,  cannyLowThreshold,  cannyHighThreshold,  kernel_size,  houghThreshold,  min_line_length = 50,  max_line_gap = 50): 
    blurredImage = cv2.GaussianBlur(img[:, :, 2], (kernel_size, kernel_size), 0)
    Edges = cv2.Canny(blurredImage, cannyLowThreshold, cannyHighThreshold)
    if(len(Edges)>0):
        lines_CurrentWorkingImage, computedLines = hough_lines(Edges, houghThreshold, min_line_length, max_line_gap)
    
        #replicate channel 2
        channel2Image = np.zeros_like(img)
        channel2Image[:, :, 0] = img[:, :, 2]
        channel2Image[:, :, 1] = img[:, :, 2]
        channel2Image[:, :, 2] = img[:, :, 2]
        
        
        #This will add lines on channel 0
        superPosedImageWithBrokenLines = weighted_img(lines_CurrentWorkingImage, channel2Image, α=0.8, β=1., λ=0.)
        
        superPosedImageWithBrokenLines[:, :, 2] = superPosedImageWithBrokenLines[:, :, 0]
        superPosedImageWithBrokenLines[:, :, 0] = channel2Image[:, :, 0]
        superPosedImageWithBrokenLines[:, :, 1] = channel2Image[:, :, 1]
    
    else:
        return img
    
    return superPosedImageWithBrokenLines
    

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img[:, :, 2]
    returnedImage = np.zeros_like(img)

    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dirGradients = np.arctan2(abs_sobel_y, abs_sobel_x) 
    #maxGradient = np.max(dirGradients)
    # 5) Create a binary mask where direction thresholds are met
    dirGradientsbinary = np.zeros_like(dirGradients)
    dirGradientsbinary[(dirGradients >= thresh[0]) & (dirGradients <= thresh[1])] = 255
    
    returnedImage[:, :, 0] = dirGradientsbinary
    returnedImage[:, :, 1] = dirGradientsbinary
    returnedImage[:, :, 2] = dirGradientsbinary
    

    return returnedImage    
    

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True: # Call with two outputs if vis==True to visualize the HOG
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      # Otherwise call with one output
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
        
        

def warp(image, camera_matrix,  dist_coefs):
    srcPoints = [
                #bottom left
                 [249, 690], 
                 #top left
                 [605, 443],
                #top right`
                [674,  443],
                #bottom right
                [1059,  690]
                 ]
    
    
    dstPoints = [
                #bottom left
                 [249, 690], 
                 #top left
                 [249, 0],
                #top right`
                [1059,  0],
                #bottom right
                [1059,  690]
                 ]
                 
    h, w = image.shape[1],  image.shape[0]             
    M = cv2.getPerspectiveTransform(np.asarray(dstPoints).astype(np.float32), np.asarray(srcPoints).astype(np.float32))             
    unwarpedReferenceImage =  cv2.warpPerspective(image, M,  (int(1*h),  int(1*w)),  flags=cv2.INTER_LINEAR) 
    return unwarpedReferenceImage
    
    
def unwarp(image, camera_matrix,  dist_coefs):
    
    #Undistort the image
    
    dstColor = cv2.undistort(image, camera_matrix, dist_coefs, None, camera_matrix)
    
    #plt.imshow(dstColor)
    #plt.show()
    
    srcPoints = [
                #bottom left
                 [249, 690], 
                 #top left
                 [605, 443],
                #top right`
                [674,  443],
                #bottom right
                [1059,  690]
                 ]
    
    
    dstPoints = [
                #bottom left
                 [249, 690], 
                 #top left
                 [249, 0],
                #top right`
                [1059,  0],
                #bottom right
                [1059,  690]
                 ]
    
    M = cv2.getPerspectiveTransform(np.asarray(srcPoints).astype(np.float32), np.asarray(dstPoints).astype(np.float32))
    
    h, w = image.shape[1],  image.shape[0]
    
    unwarpedImage =  cv2.warpPerspective(dstColor, M,  (int(1*h),  int(1*w)),  flags=cv2.INTER_LINEAR) 
    return unwarpedImage


def doHOG(image,  orientations,  pixels_per_cell,  cells_per_block,  visualise):
    #['image', 'orientations', 'pixels_per_cell', 'cells_per_block', 'visualise', 'transform_sqrt', 'feature_vector', 'normalise']
    returnedImage = np.zeros_like(image)
    
    fd, hog_image = hog(image[:, :, 0], orientations=orientations , pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=visualise)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    
    
    returnedImage[:, :, 0] = (hog_image_rescaled*255).astype(np.uint8)
    returnedImage[:, :, 1] = (hog_image_rescaled*255).astype(np.uint8)
    returnedImage[:, :, 2] = (hog_image_rescaled*255).astype(np.uint8)
    
    #print("Returning from HOG...")
    
    return returnedImage


def histogramBasedLaneMarking(image):
    
    markedImage = np.copy(image)
    
    #unwarped image is expected.
    histogram = np.sum(image[(image.shape[0]-configuration.imageOffsetFromTop)//2:,:], axis=0)
    
    #mid point of the image
    midpoint = np.int(histogram.shape[0]/2)
    
    # Set height of windows
    window_height = np.int((image.shape[0]-configuration.imageOffsetFromTop)/configuration.nHistogramWindows)
    
    #Compute center-of-gravity for each window
    #Left Half
    leftHalfCentersofGravity = []
    leftX = []
    leftY = []
    for currentWindowIndex in range(configuration.nHistogramWindows):
        #Use only channel 0
        windowOfInterest = image[configuration.imageOffsetFromTop+currentWindowIndex*window_height:configuration.imageOffsetFromTop+(currentWindowIndex+1)*window_height,0:midpoint-150,0]
        if(np.isfinite(windowOfInterest).all() and np.any(windowOfInterest)):
            try:
                relativeCenterOfGravity = ndimage.measurements.center_of_mass(windowOfInterest,[1],[1])
            except:
                print("Problem Here")
                                                                      
            #Convert to absolute coordinates; top-left corner of the window is relative 0,0
            if(not (np.isnan(relativeCenterOfGravity)).any()):
                #absoluteCenterOfGravity = [relativeCenterOfGravity[0][0]+0,  relativeCenterOfGravity[0][1]+currentWindowIndex*window_height]
                leftX.append(int(relativeCenterOfGravity[0][1]+0))
                leftY.append(int(relativeCenterOfGravity[0][0]+currentWindowIndex*window_height + configuration.imageOffsetFromTop))
                absoluteCenterOfGravity = (int(relativeCenterOfGravity[0][0]+currentWindowIndex*window_height+configuration.imageOffsetFromTop) , int(relativeCenterOfGravity[0][1]+0))
                leftHalfCentersofGravity.append(absoluteCenterOfGravity)
        
        
    #Right Half
    rightHalfCentersofGravity = []
    rightX = []
    rightY = []
    for currentWindowIndex in range(configuration.nHistogramWindows):
        #Use only channel 0
        windowOfInterest = image[configuration.imageOffsetFromTop+ currentWindowIndex*window_height: configuration.imageOffsetFromTop+(currentWindowIndex+1)*window_height,midpoint+150:, 0]
        if(np.isfinite(windowOfInterest).all() and np.any(windowOfInterest)):
            relativeCenterOfGravity = ndimage.measurements.center_of_mass(windowOfInterest,[1],[1])
            
            if (not (np.isnan(relativeCenterOfGravity)).any()):
                #Convert to absolute coordinates; top-left corner of the window is relative 0,0
                rightX.append(int(relativeCenterOfGravity[0][1]+midpoint+150))
                rightY.append(int(relativeCenterOfGravity[0][0]+currentWindowIndex*window_height + configuration.imageOffsetFromTop))
                
                absoluteCenterOfGravity = (int(relativeCenterOfGravity[0][0]+currentWindowIndex*window_height+configuration.imageOffsetFromTop),  int(relativeCenterOfGravity[0][1]+midpoint+150))
                rightHalfCentersofGravity.append(absoluteCenterOfGravity)
    
    
    for point in leftHalfCentersofGravity:
        cv2.circle(markedImage,(point[1], point[0]), 10, (255,0, 255),  -1)
    
    for point in rightHalfCentersofGravity:
        cv2.circle(markedImage,(point[1], point[0]), 10, (0, 255, 255),  -1)
        
    
    markedImage = fitLines(markedImage, leftHalfCentersofGravity,  rightHalfCentersofGravity,  leftX, leftY, rightX,  rightY)
    
    return markedImage,  leftHalfCentersofGravity, rightHalfCentersofGravity



def fitLines(markedImage,  leftHalfCentersofGravity,  rightHalfCentersofGravity,  leftX, leftY, rightX,  rightY):
    
    #Try Curve Fitting
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
    
    #Clean data by removing outliers
    #http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    #http://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm#MAD
    #https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    
    #Eliminate outliers based on more than acceptable differences in slope. Points are eliminates based on relative changes with respect to neighboring points
    #compute slopes
    
    if(len(leftHalfCentersofGravity) == 0 or len(rightHalfCentersofGravity)==0):
        print("No left or right lanes found. Abort")
        sys.exit()
    
    leftSideSlopes = []
    for leftSideCoordinatesIndex in range(len(leftHalfCentersofGravity)-1):
    #for leftSideCoordinatesIndex in range(len(leftHalfCentersofGravity)):
        deltaY = leftHalfCentersofGravity[leftSideCoordinatesIndex+1][0]-leftHalfCentersofGravity[leftSideCoordinatesIndex][0]
        deltaX = leftHalfCentersofGravity[leftSideCoordinatesIndex+1][1]-leftHalfCentersofGravity[leftSideCoordinatesIndex][1]
        #deltaY = leftHalfCentersofGravity[leftSideCoordinatesIndex][0]
        #deltaX = leftHalfCentersofGravity[leftSideCoordinatesIndex][1]
        
        #skip the point if
        if(deltaX is not 0):
            slope = deltaY/(deltaX*1.)
        else:
            slope = float('inf')
            
        degrees = np.rad2deg(np.arctan(slope))
        if(degrees < 0):
            degrees = 180+degrees
            
        leftSideSlopes.append(degrees)
    
    #compute difference between successive slopes
    deltaSlopes = []
    for slopeIndex in range(len(leftSideSlopes)-1):
        deltaSlopes.append(leftSideSlopes[slopeIndex+1] - leftSideSlopes[slopeIndex])
    
    #Eliminate
    deviationFromMedian = leftSideSlopes - np.median(np.sort(leftSideSlopes))
    outliers= 0.6745*np.abs(deviationFromMedian)/(np.median(np.sort(np.abs(deviationFromMedian))))
    #inliers = outliers<3.5*0.6745
    inliers = outliers<1.5
    #inliers = (deviationFromMedian<0.1*np.median(np.sort(leftSideSlopes))) | (outliers<1.5)
    #leftSideSlopes = np.array(leftSideSlopes)[inliers==True]
    
    #Clean the corresponding coordinates
    #The list of inliers is at max len(Array)-1 because of slopes
    #Prepend default "True" so that the inliers list is as long as the number of coordinates
    #inliers = np.append(True,  inliers)
    #leftHalfCentersofGravity = np.array(leftHalfCentersofGravity)[inliers==True]
    
    leftHalfCentersofGravity = np.copy(np.asarray(leftHalfCentersofGravity))
    inliersCopy = np.copy(inliers)
    for index, inlier in enumerate(inliersCopy):
        if(inliersCopy[0]==False):
            inliersCopy = np.delete(inliersCopy, 0)
            leftHalfCentersofGravity = np.delete(leftHalfCentersofGravity, 0, 0)
            continue
            
        if(inliersCopy[-1]==False):
            inliersCopy = np.delete(inliersCopy, -1)
            np.delete(leftHalfCentersofGravity, -1, 0)
            continue
        
        if(index<len(inliersCopy) and inliersCopy[index]==False and inliersCopy[index+1]==False):
            inliersCopy= np.delete(inliersCopy,  index)
            inliersCopy = np.delete(inliersCopy,  index)
            np.delete(leftHalfCentersofGravity,  index+1,  0)
            continue
        
        if(index<len(inliersCopy) and inliersCopy[index]==False and inliersCopy[index+1]==True):
            inliersCopy = np.delete(inliersCopy,  index)
            inliersCopy = np.delete(inliersCopy,  index)
            np.delete(leftHalfCentersofGravity,  index+1,  0)
            continue
    
    
    
    #Right Side
    rightSideSlopes = []
    for rightSideCoordinatesIndex in range(len(rightHalfCentersofGravity)-1):
    #for rightSideCoordinatesIndex in range(len(rightHalfCentersofGravity)):
        deltaY = rightHalfCentersofGravity[rightSideCoordinatesIndex+1][0]-rightHalfCentersofGravity[rightSideCoordinatesIndex][0]
        deltaX = rightHalfCentersofGravity[rightSideCoordinatesIndex+1][1]-rightHalfCentersofGravity[rightSideCoordinatesIndex][1]
        #deltaY = rightHalfCentersofGravity[rightSideCoordinatesIndex][0]
        #deltaX = rightHalfCentersofGravity[rightSideCoordinatesIndex][1]
    
        #skip the point if
        if(deltaX is not 0):
            slope = deltaY/(deltaX*1.)
        else:
            slope = float('inf')
            
        
        degrees = np.rad2deg(np.arctan(slope))
        if(degrees < 0):
            degrees = 180+degrees
            
        rightSideSlopes.append(degrees)
        
    
    #Eliminate
    deviationFromMedian = rightSideSlopes - np.median(np.sort(rightSideSlopes))
    outliers= 0.6745*np.abs(deviationFromMedian)/(np.median(np.abs(deviationFromMedian)))
    #inliers = outliers<3.5*0.6745
    inliers = outliers<1.5
    #inliers = (deviationFromMedian<0.1*np.median(np.sort(rightSideSlopes))) | (outliers<1.5)
    
    #Duplicate the first point so that rejection of points can be based directly on the indices of inliers
    #rightHalfCentersofGravity.insert(0,rightHalfCentersofGravity[0])
    
    #rightSideSlopes = np.array(rightSideSlopes)[inliers==True]
    #Clean the corresponding coordinates
    #The list of inliers is at max len(Array)-1 because of slopes
    #Prepend default "True" so that the inliers list is as long as the number of coordinates
    #inliers = np.append(True,  inliers)
    #Reject coordinates based on the following truth table
    rightHalfCentersofGravity = np.copy(np.asarray(rightHalfCentersofGravity))
    inliersCopy = np.copy(inliers)
    for index, inlier in enumerate(inliersCopy):
        if(inliersCopy[0]==False):
            inliersCopy = np.delete(inliersCopy, 0)
            rightHalfCentersofGravity = np.delete(rightHalfCentersofGravity, 0, 0)
            continue
            
        if(inliersCopy[-1]==False):
            inliersCopy = np.delete(inliersCopy, -1)
            np.delete(rightHalfCentersofGravity, -1, 0)
            continue
        
        if(index<len(inliersCopy) and inliersCopy[index]==False and inliersCopy[index+1]==False):
            inliersCopy= np.delete(inliersCopy,  index)
            
            #Because index + 1 is now index after the first delete
            inliersCopy = np.delete(inliersCopy,  index)
            np.delete(rightHalfCentersofGravity,  index+1,  0)
            continue
        
        if(index<len(inliersCopy) and inliersCopy[index]==False and inliersCopy[index+1]==True):
            inliersCopy = np.delete(inliersCopy,  index)
            inliersCopy = np.delete(inliersCopy,  index)
            np.delete(rightHalfCentersofGravity,  index+1,  0)
            continue
    
    
    
    
    
    #rightHalfCentersofGravity = np.array(rightHalfCentersofGravity)[inliers==True]
    
    #Remove the duplicate
    #rightHalfCentersofGravity = np.array(rightHalfCentersofGravity)[1:,]
    
    
    markedImageWithLines = np.copy(markedImage)
    
    #Mark the observation which has more data points
    if(len(leftHalfCentersofGravity)>len(rightHalfCentersofGravity) and (len(leftHalfCentersofGravity)>=2)):
        #We have higher confidence in the left detected border
        
        
        #Keep Y as an independent axis
        leftLanePoly = np.poly1d(np.polyfit(leftHalfCentersofGravity[:, 0], leftHalfCentersofGravity[:, 1], 2))
        
        #We ignore the right detected corridor when there are too few points detected
        #if(len(rightHalfCentersofGravity)< int(len(leftHalfCentersofGravity)/2)):
        if(len(rightHalfCentersofGravity)< 3):
            #We will use the left detected corridor to mark the right corridor
            
            #Find the difference between lanes
            minPoints = np.minimum(len(leftHalfCentersofGravity[:, 1]), len(rightHalfCentersofGravity[:, 1]))
            distanceBetweenLanes = np.sqrt(((np.array(rightHalfCentersofGravity[:, 1][:minPoints]) - np.array(leftHalfCentersofGravity[:, 1][:minPoints])) ** 2).mean())
            
            #Draw left border
            yLeftLine = np.linspace(np.maximum(leftHalfCentersofGravity[:, 0][-1], 650), np.minimum(leftHalfCentersofGravity[:, 0][0], 50), 10)
            #Get the corresponding X-coordinates
            xLeftLine = leftLanePoly(yLeftLine)
            
            #Get the right border from the left border
            yRightLine = np.copy(yLeftLine)
            xRightLine = xLeftLine+distanceBetweenLanes
            
        else:
            rightLanePoly = np.poly1d(np.polyfit(leftHalfCentersofGravity[:, 1], leftHalfCentersofGravity[:, 0], 2))
        
            #Both left and right borders can be painted independently
            
            #Left Border
            leftLanePoly = np.poly1d(np.polyfit(leftHalfCentersofGravity[:, 0], leftHalfCentersofGravity[:, 1], 2))
            #Draw left border
            yLeftLine = np.linspace(np.maximum(leftHalfCentersofGravity[:, 0][-1], 650), np.minimum(leftHalfCentersofGravity[:, 0][0], 50), 10)
            #Get the corresponding X-coordinates
            xLeftLine = leftLanePoly(yLeftLine)
            
            #Draw right border
            rightLanePoly = np.poly1d(np.polyfit(rightHalfCentersofGravity[:, 0], rightHalfCentersofGravity[:, 1], 2))
            
            yRightLine = np.linspace(np.maximum(rightHalfCentersofGravity[:, 0][-1], 650), np.minimum(rightHalfCentersofGravity[:, 0][0], 50), 10)
            #Get the corresponding X-coordinates
            xRightLine = rightLanePoly(yRightLine)
    
    
        #Mark the borders
        #Left Side
        for index in range(len(yLeftLine)-1):
            cv2.line(markedImageWithLines, ( int(xLeftLine[index]), int(yLeftLine[index])), ( int(xLeftLine[index+1]), int(yLeftLine[index+1])), (255,0, 255), 10)
        
        #Right Side
        for index in range(len(yRightLine)-1):
            cv2.line(markedImageWithLines, ( int(xRightLine[index]), int(yRightLine[index])), ( int(xRightLine[index+1]), int(yRightLine[index+1])), (0, 255, 255), 10)
        
        
        
    elif(len(rightHalfCentersofGravity[:, 0])>=len(leftHalfCentersofGravity[:, 0]) and (len(rightHalfCentersofGravity[:, 0])>=2)):
        #We have higher confidence in the right detected lane
        #Keep Y as an independent axis
        rightLanePoly = np.poly1d(np.polyfit(rightHalfCentersofGravity[:, 0], rightHalfCentersofGravity[:, 1], 2))
        
        #We ignore the left detected corridor when there are too few points detected
        #if(len(leftHalfCentersofGravity[:, 0])< int(len(rightHalfCentersofGravity[:, 0])/2)):
        if(len(leftHalfCentersofGravity[:, 0])< 3):
            #We will use the right detected corridor to mark the left corridor
            
            #Find the difference between lanes
            minPoints = np.minimum(len(leftHalfCentersofGravity[:, 1]), len(rightHalfCentersofGravity[:, 1]))
            distanceBetweenLanes = np.sqrt(((np.array(rightHalfCentersofGravity[:, 1][:minPoints]) - np.array(leftHalfCentersofGravity[:, 1][:minPoints])) ** 2).mean())
            
            #Draw right border
            yRightLine = np.linspace(np.maximum(rightHalfCentersofGravity[:, 0][-1], 650), np.minimum(rightHalfCentersofGravity[:, 0][0], 50), 10)
            #Get the corresponding X-coordinates
            xRightLine = rightLanePoly(yRightLine)
            
            #Get the left border from the right border
            yLeftLine = np.copy(yRightLine)
            xLeftLine = xRightLine-distanceBetweenLanes
            
        else:
            #rightLanePoly = np.poly1d(np.polyfit(leftHalfCentersofGravity[:, 1], leftHalfCentersofGravity[:, 0], 2))
        
            #Both left and right borders can be painted independently
            
            #Left Border
            leftLanePoly = np.poly1d(np.polyfit(leftHalfCentersofGravity[:, 0], leftHalfCentersofGravity[:, 1], 2))
            #Draw left border
            yLeftLine = np.linspace(np.maximum(leftHalfCentersofGravity[:, 0][-1], 650), np.minimum(leftHalfCentersofGravity[:, 0][0], 50), 10)
            #Get the corresponding X-coordinates
            xLeftLine = leftLanePoly(yLeftLine)
            
            #Draw right border
            rightLanePoly = np.poly1d(np.polyfit(rightHalfCentersofGravity[:, 0], rightHalfCentersofGravity[:, 1], 2))
            
            yRightLine = np.linspace(np.maximum(rightHalfCentersofGravity[:, 0][-1], 650), np.minimum(rightHalfCentersofGravity[:, 0][0], 50), 10)
            #Get the corresponding X-coordinates
            xRightLine = rightLanePoly(yRightLine)
            
            
        #Mark the borders
        #Left Side
        for index in range(len(yLeftLine)-1):
            cv2.line(markedImageWithLines, ( int(xLeftLine[index]), int(yLeftLine[index])), ( int(xLeftLine[index+1]), int(yLeftLine[index+1])), (255,0, 255), 10)
        
        #Right Side
        for index in range(len(yRightLine)-1):
            cv2.line(markedImageWithLines, ( int(xRightLine[index]), int(yRightLine[index])), ( int(xRightLine[index+1]), int(yRightLine[index+1])), (0, 255, 255), 10)
        
        
        
        
    else:
        #Not enough confidence to paint either border
        print("Did not fit any polynomial",  end='')
        print("len(leftY): %d, len(rightY): %d" % (len(leftHalfCentersofGravity[:, 0]),  len(rightHalfCentersofGravity[:, 0]) ))
        sys.exit()
        
    #Add the lane color
    pts_left = np.array([np.transpose(np.vstack([xLeftLine, yLeftLine]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([xRightLine, yRightLine])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(markedImageWithLines, np.int_([pts]), (0,255, 0))
        
    return markedImageWithLines
    
    

