# -*- coding: utf-8 -*-

# Standard Libraries
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from multiprocessing import Pool
import cv2
from functools import partial
import traceback
import sys
import skvideo.io

# Configuration Settings
import configuration

# Custom Helper functions
from LaneDetectionUtils import clipImages,  normalizeImages,  changeColorSpace,  isolatePixelsPerChannel,  histogramBasedLaneMarking
from LaneDetectionUtils import customContrast,  customIsolatePixel, doEdgeDetection,  dir_threshold,  getCalibrationCoefficients,  unwarp,  warp
from LaneDetectionUtils import doHOG



                
            
# Worker Functions
def doLaneDetection():

    #Read the contents of the input test_image directory
    imageFiles = os.listdir(configuration.locationImageFiles)
    
    f, axes = plt.subplots(3, 2, figsize=(2*7.2, 2*12.8))
    f.tight_layout()
    plt.ioff()
    #plt.ion()
    
    #one time-setting a sample image
    fullPathtoCurrentImage = os.path.join(configuration.locationImageFiles, imageFiles[0])
    filename, file_extension = os.path.splitext(fullPathtoCurrentImage)
            
    #Read the image
    if((file_extension == ".png") or (file_extension == ".jpeg")):
                currentImageForProcessing = mpimg.imread(fullPathtoCurrentImage)
            
    # Change the range of pixel values to match those in the case of a jpeg image
    if(file_extension == ".png"):
        currentImageForProcessing = (currentImageForProcessing*255).astype(np.uint8)
                
    #Channel 0
    ax00 = axes[0, 0].imshow(currentImageForProcessing[:, :, 0],  cmap='gray')
    axes[0, 0].set_title('Channel 0', fontsize=10)
    
    #Channel 1
    ax10 = axes[1, 0].imshow(currentImageForProcessing[:, :, 1],  cmap='gray')
    axes[1, 0].set_title('Channel 1', fontsize=10)
    
    #Channel 2
    ax20 = axes[2, 0].imshow(currentImageForProcessing[:, :, 2],  cmap='gray')
    axes[2, 0].set_title('Channel 2', fontsize=10)
    
    #Combined
    ax01 = axes[0, 1].imshow(currentImageForProcessing)
    axes[0, 1].set_title('Combined', fontsize=10)
    
    #Original
    ax11 = axes[1, 1].imshow(currentImageForProcessing)
    axes[1, 1].set_title('Original', fontsize=10)

    

    while(True):
        
        #For each image...
        images = []
        imageNames = []

        for imageIndex in range(len(imageFiles)):
        
            #Get full path to each image that we would like to read
            fullPathtoCurrentImage = os.path.join(configuration.locationImageFiles, imageFiles[imageIndex])
            
            #Get the file extension
            filename, file_extension = os.path.splitext(fullPathtoCurrentImage)
            
            #Read the image
            if((file_extension == ".png") or (file_extension == ".jpeg")):
                currentImageForProcessing = mpimg.imread(fullPathtoCurrentImage)
            
            # Change the range of pixel values to match those in the case of a jpeg image
            if(file_extension == ".png"):
                currentImageForProcessing = (currentImageForProcessing*255).astype(np.uint8)
            
            images.append(currentImageForProcessing)
            imageNames.append(filename)
            
            # Multiprocess, with number of threads limited to nProcesses
            if((len(images)==configuration.nProcesses) or (imageIndex+1==len(imageFiles))):
                
                nThreadsRequired = min(configuration.nProcesses,  len(images))
                pool = Pool( processes=nThreadsRequired )
                                
                #order: pool.imap(partial(functionName,SecondArgument),firstArgument)
                #https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-maultiple-arguments
                for combinedImages,name in pool.starmap(processImages, zip(images,imageNames)) :

                        finalImage = combinedImages[0]
                        
                        # Set channel 0
                        ax00.set_data(finalImage[:, :, 0])
                        
                        
                        #Channel 1
                        ax10.set_data(finalImage[:, :, 1])
                        
                        
                        #Channel 2
                        ax20.set_data(finalImage[:, :, 2])
                        
                        
                        #Combined: Overlay the markers on top of the original image
                        overlayImage = cv2.addWeighted(combinedImages[1],1, combinedImages[0], 1, 0)
                        
                        ax01.set_data(overlayImage)
                        ax11.set_data(combinedImages[2])

                        print("Displaying " + name)
                        plt.pause(0.01)
                        
                pool.close()
                pool.join()
                
                #empty the array
                images[:] = [] 
                imageNames[:] = []

    
'''
processImages is the routine called be each thread to compute lane 
markers for each image passed to it.
'''            
def processImages(currentImageForProcessing, imageName):            

    #See if images need to be clipped
    if(configuration.Clip):
        originalImage, finalImage = clipImages(np.copy(currentImageForProcessing))
        
    else:
        finalImage = currentImageForProcessing
    
    
    #Change to a new color space
    finalImage = changeColorSpace(configuration.colorSpace,  finalImage)
    
    #Improve contrast
    if(configuration.Contrast):
        finalImage = customContrast(finalImage)
    
    
    #See if images need to be normalized
    if(configuration.normalize):
        finalImage = normalizeImages(np.copy(finalImage),  configuration.ChannelstoNormalize,  configuration.GlobalNormalization)
    else:
        finalImage = finalImage
    
    
    if(configuration.doCanny):
        finalImage = doEdgeDetection(finalImage,  int(configuration.CannyLowerThreshold) ,  int(configuration.CannyUpperThreshold) ,  int(configuration.BlurKernelSize) ,  int(configuration.houghThreshold) ,  int(configuration.MinLineLength) ,  int(configuration.MaxLineGap))
        
    
    if(configuration.directionalGradient):
            finalImage = dir_threshold(finalImage, sobel_kernel= int(configuration.DirectionalGradientKernelSize), thresh=(configuration.DirectionalGradientLowerThreshold, configuration.DirectionalGradientUpperThreshold))
            
    
    #Unwarp the image -- use the calibration results computed earlier
    if(not configuration.camera_matrix):
        camera_matrix,  dist_coefs = getCalibrationCoefficients(configuration.RelativePathtoCameraMatrix)
    

    finalImage = unwarp(finalImage, camera_matrix,  dist_coefs)
    
    
    if(configuration.doHistogramBasedLaneMarking):
        try:
            finalImage, leftHalfCentersofGravity, rightHalfCentersofGravity = histogramBasedLaneMarking(finalImage)
        except:
            tb = traceback.format_exc()
            print(tb)
        else:
            pass
        finally:
            pass
            

        
        
    #Try HOG to get lanes
    #Good starting point: hog(image[:, :, 0], orientations=2 , pixels_per_cell=(50,72), cells_per_block=(1,1), visualise=visualise)
    #print("HOG with %d orientations, %d x-Pixels per cell, %d Y-pixels per cell, %d x-cells per block, %d y-cells per block" %(int(configuration.nHOGOrientations), int(configuration.nHOGPixelsPerCellX),  int(configuration.nHOGPixelsPerCellY),  int(configuration.nHOGCellsPerBlockX), int(configuration.nHOGCellsPerBlockY)))
    if(configuration.doHOG):
        finalImage = doHOG(np.copy(finalImage), int(configuration.nHOGOrientations), (int(configuration.nHOGPixelsPerCellX) , int(configuration.nHOGPixelsPerCellY)), (int(configuration.nHOGCellsPerBlockX) , int(configuration.nHOGCellsPerBlockY)), True)
        
    
    #Warp again
    finalImage = warp(finalImage, camera_matrix,  dist_coefs)
    
    
    combinedImages = []
    
    combinedImages.append(finalImage)
    #combinedImages.append(currentImageForProcessing)
    
    #Warp the original image for comparison
    overlayImage = cv2.addWeighted(finalImage,1, originalImage, 1, 0)
    combinedImages.append(overlayImage)
    
    
    #unwarpedOriginal = unwarp(originalImage, camera_matrix,  dist_coefs)
    #combinedImages.append(unwarpedOriginal)
    
    
    
    combinedImages.append(originalImage)  
    
    
    #return finalImage
    #print("Done. Shapes (Unwarped Original: %d, %d, %d" %(unwarpedOriginal.shape),  end='')
    #print(". finalImage: %d, %d, %d" %(finalImage.shape))
    return combinedImages,  imageName
        
            

def doLaneDetection_Video():
    
    #Read the contents of the input test_image directory
    imageFiles = os.listdir(configuration.locationImageFiles)
    
    f, axes = plt.subplots(1, 1, figsize=(2*7.2, 2*12.8))
    f.tight_layout()
    plt.ioff()
    
    #one time-setting a sample image
    fullPathtoCurrentImage = os.path.join(configuration.locationImageFiles, imageFiles[0])
    filename, file_extension = os.path.splitext(fullPathtoCurrentImage)
            
    #Read the image
    if((file_extension == ".png") or (file_extension == ".jpeg")):
        currentImageForProcessing = mpimg.imread(fullPathtoCurrentImage)
    
    if(file_extension == ".png"):
        currentImageForProcessing = (currentImageForProcessing*255).astype(np.uint8)
        
        
    ax01 = axes.imshow(currentImageForProcessing)
    axes.set_title('Final', fontsize=10)
    
    videoFiles = os.listdir(configuration.locationOfInputVideos)
    for video in videoFiles:
        #if(video == "challenge_video.mp4"):
        if(True):
            print("Processing: ",  video)
            
            fullPathtoInputVideo = os.path.join(configuration.locationOfInputVideos, video)
            
            #Collect metadata about the video
            metaData = skvideo.io.ffprobe(fullPathtoInputVideo)
            #Number of frames
            nFrames = int(metaData['video']['@nb_frames']) 
   

            #For writing output video
            fullPathtoOutputVideo = os.path.join(configuration.locationOfOutputVideos, "out_"+video)
            #writer = skvideo.io.FFmpegWriter(fullPathtoOutputVideo, (nFrames, int(metaData['video']['@coded_height']), int(metaData['video']['@coded_width']), 3))
            writer = skvideo.io.FFmpegWriter(fullPathtoOutputVideo, outputdict={'-vcodec': 'libx264', '-b': '750100000'})
            #writer = skvideo.io.FFmpegWriter(fullPathtoOutputVideo)
            #for i in xrange(5):
                #writer.writeFrame(outputdata[i, :, :, :])
            
            
            videoGenerator = skvideo.io.vreader(fullPathtoInputVideo)
            frameCounter = 1
            bundledFrames = []
            bundledFrameCounter = []
            for frame in videoGenerator:
                
                bundledFrames.append(frame)
                bundledFrameCounter.append('Frame: ' + str(frameCounter))
                #Update the running frame counter
                
                #Update the frame counter
                frameCounter = frameCounter + 1
                
                if(len(bundledFrames)==configuration.nProcesses or (frameCounter == nFrames)):
                    
                    nThreadsRequired = min(configuration.nProcesses,  len(bundledFrames))
                    
                    pool = Pool( processes=nThreadsRequired )
                    
                    for combinedImages,name in pool.starmap(processImages, zip(bundledFrames,bundledFrameCounter)) :
                    
                        overlayImage = cv2.addWeighted(combinedImages[1],1, combinedImages[0], 1, 0)
                        
                        ax01.set_data(overlayImage)
                        ax01.axes.set_title(name, fontsize=10)
                        
                        #plt.imshow(overlayImage)
                        #skvideo.io.vwrite(fullPathtoOutputVideo, overlayImage)
                        writer.writeFrame(overlayImage)
                        #print("Displaying " + name)
                        plt.pause(0.001)
                            
                        
                    bundledFrames[:] = []
                    bundledFrameCounter[:]= []
                    
                    pool.close()
                    pool.join()
        
                
            writer.close()


# Call the function below to mark invidiual images
doLaneDetection()

# Call the function below to mark videos
#doLaneDetection_Video()





##-------------------------- TESTING---------------------------------
#
#
##Read the contents of the input test_image directory
#imageFiles = os.listdir(configuration.locationImageFiles)
#
##For each image...
#for image in imageFiles:
#    
#    #Get full path to each image that we would like to read
#    fullPathtoCurrentImage = os.path.join(configuration.locationImageFiles, image)
#    
#    #print the full path
#    #print(fullPathtoCurrentImage)
#    
#    #Get the file extension
#    filename, file_extension = os.path.splitext(fullPathtoCurrentImage)
#    
#    #Read the image
#    currentImageForProcessing = mpimg.imread(fullPathtoCurrentImage)
#    
#    if(file_extension == ".png"):
#        currentImageForProcessing = (currentImageForProcessing*255).astype(np.uint8)
#    
#    
#    #See if images need to be clipped
#    finalImage = clipImages(np.copy(currentImageForProcessing))
#    finalImage = changeColorSpace(configuration.colorSpace,  finalImage)
#    finalImage = customContrast(finalImage)
#    finalImage = doEdgeDetection(finalImage,  int(configuration.CannyLowerThreshold) ,  int(configuration.CannyUpperThreshold) ,  int(configuration.BlurKernelSize) ,  int(configuration.houghThreshold) ,  int(configuration.MinLineLength) ,  int(configuration.MaxLineGap))    
#    finalImage = dir_threshold(finalImage, sobel_kernel= int(configuration.DirectionalGradientKernelSize), thresh=(configuration.DirectionalGradientLowerThreshold, configuration.DirectionalGradientUpperThreshold))
#    camera_matrix,  dist_coefs = getCalibrationCoefficients(configuration.RelativePathtoCameraMatrix)
#    finalImage = unwarp(finalImage, camera_matrix,  dist_coefs)
#    finalImage = doHOG(finalImage, int(configuration.nHOGOrientations), (int(configuration.nHOGPixelsPerCellX) , int(configuration.nHOGPixelsPerCellY)), (int(configuration.nHOGCellsPerBlockX) , int(configuration.nHOGCellsPerBlockY)), True)
#    
#    
#    #plt.ioff()
#    f, axes = plt.subplots(4, 1, figsize=(4.5*7.2,12.8))
#    f.tight_layout()
#    
#    #Channel 0
#    axes[0].imshow(finalImage[:, :, 0],cmap='gray')
#    axes[0].set_title('Channel 0', fontsize=20)
#    
#    #Channel 1
#    axes[1].imshow(finalImage[:, :, 1],  cmap='gray')
#    axes[1].set_title('Channel 1', fontsize=20)
#    
#    #Channel 2
#    axes[2].imshow(finalImage[:, :, 2],  cmap='gray')
#    axes[2].set_title('Channel 2', fontsize=20)
#    
#    #Combined
#    axes[3].imshow(finalImage)
#    axes[3].set_title('Combined', fontsize=20)
#    
#    #plt.imshow(finalImage)
#    plt.pause(1)
#    plt.close()
#    break
#    
#    
