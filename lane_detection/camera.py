"""Camera intrinsics and extrinsics. Input images are expected to be already undistorted."""

import numpy as np

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

CAMERA_MATRIX = np.array([
    [1000.0,    0.0, 960.0],
    [   0.0, 1000.0, 540.0],
    [   0.0,    0.0,   1.0],
])

# Ground-to-camera transform, X_cam = R @ X_ground + t, as produced by solvePnP in
# vehicle_calibration/calibrate_extrinsics.py: ground plane is Z=0, x right, y forward.
ROTATION_MATRIX = np.array([
    [0.99572310, -0.08471567,  0.03686142],
    [0.02363038, -0.15218366, -0.98806970],
    [0.08931469,  0.98471488, -0.14953092],
])
TRANSLATION = np.array([-0.12956601, 0.71135492, 2.87100553])
