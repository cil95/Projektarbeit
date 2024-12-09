import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

def calibrate(showPics=True):
    # Read Images
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'demoImages/calibration')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    # Initialize
    nRows = 9
    nCols = 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []

    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)
            if showPics:
                cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(500)
    cv.destroyAllWindows()

    # Calibrate
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imgGray.shape[::-1], None, None
    )
    print('Camera Matrix:\n', camMatrix)
    print('Distortion Coefficients:\n', distCoeff)

    # Save the calibration results
    np.savez('calibration_data.npz', camMatrix=camMatrix, distCoeff=distCoeff, rvecs=rvecs, tvecs=tvecs)

    return camMatrix, distCoeff

if __name__ == "__main__":
    calibrate()