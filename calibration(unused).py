import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
from settings import CALIBRATION_FILE_NAME

# def calibrate(filename, silent = True):
    
#     # pickle the data and save it
#     # calibration_data = {'camera_matrix':mtx,
#     #                     'dist_coeffs':dist,
#     #                     'img_size':img_size}
#     # with open(filename, 'wb') as f:
#     #     pickle.dump(calibration_data, f)

#     # if not silent:
#     #     for image_file in os.listdir(images_path):
#     #         if image_file.endswith("jpg"):
#     #             # show distorted images
#     #             img = mpimg.imread(os.path.join(images_path, image_file))
#     #             plt.imshow(cv2.undistort(img, mtx, dist))
#     #             plt.show()

#     # return mtx, dist

# if __name__ == '__main__':
#     calibrate(CALIBRATION_FILE_NAME, True)
# images_path = '\camera_cal'
chessboardSize = ()
n_x = 9
n_y = 6
# setup object points
objp = np.zeros((n_y*n_x, 3), np.float32)
objp[:, :2] = np.mgrid[0:n_x, 0:n_y].T.reshape(-1, 2)
image_points = []
object_points = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# loop through provided images
for image_file in os.listdir(images_path) :
    if image_file.endswith("jpg") :
        # turn images to grayscale and find chessboard corners
        img = mpimg.imread(os.path.join(images_path, image_file))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(img_gray, (n_x, n_y))
        if ret == True:
            # make fine adjustments to the corners so higher precision can be obtained before
            # appending them to the list
            object_points.append(objp)
            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1, -1), criteria)
            image_points.append(corners2)
            
            # Draw and display the corners 
            cv2.drawChessboardCorners(img, (n_x, n_y), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1000)
            # if not silent:
            #     plt.imshow(img)
            #     plt.show()
cv2.destroyAllWindows()

# perform the calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_gray.shape[::-1], None, None)
img_size = img.shape
print("Camera Calibrated: ", ret)
print("\nCamera Matrix:\n", mtx)
print("\nDistortion Parameters:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslatin Vectors:\n", tvecs)
# Undistortion
img = cv2.imread('D:\TUGAS AKHIR DUTA\Deteksi jarak kendaraan\camera_cal\calibration1.jpg')
h, w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w,h))
# Undistort
dst = cv2.undistort(img, mtx, dist, None, newCameraMatrix)
# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('CalResult2.png', dst)
# Reprojection Error
mean_error = 0
for i in range(len(object_points)):
    image_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2)/len(image_points2)
    mean_error += error

print("\ntotal error: {}".format(mean_error/len(object_points)))
print("\n\n\n")