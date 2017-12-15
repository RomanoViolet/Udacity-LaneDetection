#------------------------------------------
# All Imports
#------------------------------------------
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
#from IPython.core.debugger import set_trace
import os
import random

#--------------- for triggering the debugger------------------
#set_trace()
#-------------------------------------------------------------
import configuration

DEBUG = configuration.DEBUG
CHESSBOARD_COLUMNS = configuration.CHESSBOARD_COLUMNS
CHESSBOARD_ROWS = configuration.CHESSBOARD_ROWS
RELATIVE_PATH_TO_CALIBRATION_IMAGES = configuration.RELATIVE_PATH_TO_CALIBRATION_IMAGES
RELATIVE_PATH_TO_OUTPUT_CALIBRATION_FOLDER = configuration.RELATIVE_PATH_TO_OUTPUT_CALIBRATION_FOLDER
NOMINAL_HEIGHT_OF_CALIBRATION_IMAGES = configuration.NOMINAL_HEIGHT_OF_CALIBRATION_IMAGES
NOMINAL_WIDTH_OF_CALIBRATION_IMAGES = configuration.NOMINAL_WIDTH_OF_CALIBRATION_IMAGES


# Camera Calibration
#---------------------------------------------
def calibrateCamera(relPathtoCalibrationImages=RELATIVE_PATH_TO_CALIBRATION_IMAGES, relPathtoOutput=RELATIVE_PATH_TO_OUTPUT_CALIBRATION_FOLDER):
    #Create an output directory
    if not os.path.exists(relPathtoOutput):
        os.makedirs(relPathtoOutput)

    #Read the contents of the input test_image directory
    imageFiles = os.listdir(relPathtoCalibrationImages)
    
    #Randomly choose one image to use for final test
    indexOfTestImage = random.randint(0, len(imageFiles)-1)
    
    #set_trace()
    
    #Copy this image from the collected set
    testImage = imageFiles[indexOfTestImage]
    
    #For reference get the height and width of the the test image, which should
    #be the same for all image
    fullPathtoTestImage = os.path.join(relPathtoCalibrationImages, testImage)
    testImage = cv2.imread(fullPathtoTestImage)
        
    
    
    #Remove the selected image from the collected list
    #del imageFiles[indexOfTestImage]
    
    if(DEBUG):
        print("Selected Random Image: ", testImage)
        print("Calibration Images: ")
        for testimage in imageFiles:
            print(testimage)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plan

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_ROWS*CHESSBOARD_COLUMNS,3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_COLUMNS, 0:CHESSBOARD_ROWS].T.reshape(-1,2)
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(imageFiles):
        fullPathtoCurrentImage = os.path.join(relPathtoCalibrationImages, fname)
        img = cv2.imread(fullPathtoCurrentImage)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        #Abort if any image comes from a camera different to the one which produced the test sample
        
        if((gray.shape[0]>=NOMINAL_HEIGHT_OF_CALIBRATION_IMAGES) and (gray.shape[1]>=NOMINAL_WIDTH_OF_CALIBRATION_IMAGES)):
            #Image is bigger: can crop
            # Crop from x, y, w, h -> 100, 200, 300, 400
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            gray = gray[0:NOMINAL_HEIGHT_OF_CALIBRATION_IMAGES, 0:NOMINAL_WIDTH_OF_CALIBRATION_IMAGES] 
            
        else:
            assert(testImage.shape[:2] == (NOMINAL_HEIGHT_OF_CALIBRATION_IMAGES, NOMINAL_WIDTH_OF_CALIBRATION_IMAGES))
        
       
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_COLUMNS,CHESSBOARD_ROWS), None)

        # If found, add object points, image points
        if ret == True:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (CHESSBOARD_COLUMNS,CHESSBOARD_ROWS), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            if(DEBUG):
                cv2.imshow('img', img)
                cv2.waitKey(500)

    if(DEBUG):
        cv2.destroyAllWindows()
        
        
    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (NOMINAL_WIDTH_OF_CALIBRATION_IMAGES, NOMINAL_HEIGHT_OF_CALIBRATION_IMAGES), None, None)
    
    if(DEBUG):
        print("\nRMS:", rms)
        print("camera matrix:\n", camera_matrix)
        print("distortion coefficients: ", dist_coefs.ravel())
        print('')
        
        
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (NOMINAL_WIDTH_OF_CALIBRATION_IMAGES, NOMINAL_HEIGHT_OF_CALIBRATION_IMAGES), 1, (NOMINAL_WIDTH_OF_CALIBRATION_IMAGES, NOMINAL_HEIGHT_OF_CALIBRATION_IMAGES))
    
    if(DEBUG):
        print(roi)


    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["objpoints"] = objpoints
    dist_pickle["imgpoints"] = imgpoints
    dist_pickle["mtx"] = camera_matrix
    dist_pickle["dist"] = dist_coefs
    
    #Write path
    fullPathtoCameraCoeffs = os.path.join(relPathtoOutput, "wide_dist_pickle.p")
    pickle.dump( dist_pickle, open( fullPathtoCameraCoeffs, "wb" ) )
    
    # Visualize undistortion
    if(DEBUG):
        # Test undistortion on an image
        for image in imageFiles:
            fullPathtoTestImage = os.path.join(relPathtoCalibrationImages, image)
            img = cv2.imread(fullPathtoTestImage)
            #img_size = (img.shape[1], img.shape[0])
            h,  w = img.shape[:2]
            
            
            if(np.asarray(roi).all() == False):
                newcameramtx = camera_matrix
                dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
            else:
                dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
                
                # crop and save the image
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]

            
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
                ax1.imshow(img)
                ax1.set_title('Original Image', fontsize=30)
                ax2.imshow(dst)
                ax2.set_title('Undistorted Image', fontsize=30)
                plt.show()
            
            #Write Output Undistorted Image
            fullPathtoWriteUndistortedImage = os.path.join(relPathtoOutput, image)
            cv2.imwrite(fullPathtoWriteUndistortedImage,dst)

            #cv2.destroyAllWindows()    
        
        
    return objpoints, imgpoints, camera_matrix, dist_coefs

    

(objpoints, imgpoints, camera_matrix, dist_coefs) = calibrateCamera()
print("Done calibrate camera")

    


