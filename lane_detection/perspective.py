### STEP 2: Perspective Transform from Car Camera to Bird's Eye View ###

import cv2
import numpy as np

# Source points taken from images with straight lane lines, these are to become parallel after the warp transform
SRC_POINTS = np.float32([
    (190, 720),   # bottom-left corner
    (596, 447),   # top-left corner
    (685, 447),   # top-right corner
    (1125, 720),  # bottom-right corner
])
DST_OFFSET = 300


def warp(img):
    img_size = (img.shape[1], img.shape[0])

    # Destination points are to be parallel, taken into account the image size
    dst = np.float32([
        [DST_OFFSET, img_size[1]],                # bottom-left corner
        [DST_OFFSET, 0],                          # top-left corner
        [img_size[0] - DST_OFFSET, 0],            # top-right corner
        [img_size[0] - DST_OFFSET, img_size[1]],  # bottom-right corner
    ])
    # Calculate the transformation matrix and it's inverse transformation
    M = cv2.getPerspectiveTransform(SRC_POINTS, dst)
    M_inv = cv2.getPerspectiveTransform(dst, SRC_POINTS)
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M_inv
