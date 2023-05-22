import cv2 as cv
import numpy as np

def nightfilter(src_image):

    dist_image = cv.equalizeHist(src_image)
    dist_image = cv.fastNlMeansDenoising(dist_image, None, 1e6, 3, 3)
    dist_image = cv.GaussianBlur(dist_image, (3,3), 0)
    dist_image = cv.medianBlur(dist_image, 3)

    average = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16

    dist_image = cv.filter2D(dist_image, -1, average)

    enhancer = 0.9*np.array([ [-1, -2, -1],
                                [-2,  13, -2],
                                [-1, -2, -1]])

    return cv.filter2D(dist_image, -1, enhancer)