### STEP 2: Perspective Transform from Car Camera to Bird's Eye View ###

import cv2
import numpy as np

DST_OFFSET = 300

# Ground rectangle (metres) that is warped to the bird's eye view: centred on the camera,
# ROI_WIDTH wide, from ROI_NEAR to ROI_FAR ahead of it
DEFAULT_ROI_WIDTH = 3.7
DEFAULT_ROI_NEAR = 3.0
DEFAULT_ROI_FAR = 33.0


class PerspectiveTransform:
    """Undistortion and bird's eye warp for one camera.

    Extrinsics are ground-to-camera, X_cam = R @ X_ground + t, with the ground plane at Z=0,
    x to the right and y forward (as written by vehicle_calibration/calibrate_extrinsics.py).
    """

    def __init__(self, img_size, camera_matrix, dist_coeffs, rotation_matrix, translation,
                 roi_width=DEFAULT_ROI_WIDTH, roi_near=DEFAULT_ROI_NEAR, roi_far=DEFAULT_ROI_FAR):
        if roi_width <= 0 or not 0 <= roi_near < roi_far:
            raise ValueError('expected roi_width > 0 and 0 <= roi_near < roi_far')
        mtx = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        R = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
        t = np.asarray(translation, dtype=np.float64).reshape(3)
        self.img_size = img_size

        # Source points: the ground rectangle projected into the image,
        # these are to become parallel after the warp transform
        camera_position = -R.T @ t
        x_left = camera_position[0] - roi_width / 2
        x_right = camera_position[0] + roi_width / 2
        y_near = camera_position[1] + roi_near
        y_far = camera_position[1] + roi_far
        ground = np.array([
            [x_left, y_near, 0.0],   # bottom-left corner
            [x_left, y_far, 0.0],    # top-left corner
            [x_right, y_far, 0.0],   # top-right corner
            [x_right, y_near, 0.0],  # bottom-right corner
        ])
        in_camera = ground @ R.T + t
        if np.any(in_camera[:, 2] <= 0):
            raise ValueError('the region of interest is not in front of the camera')
        projected = in_camera @ mtx.T
        self.src = np.float32(projected[:, :2] / projected[:, 2:])

        # Destination points are to be parallel, taken into account the image size
        dst = np.float32([
            [DST_OFFSET, img_size[1]],                # bottom-left corner
            [DST_OFFSET, 0],                          # top-left corner
            [img_size[0] - DST_OFFSET, 0],            # top-right corner
            [img_size[0] - DST_OFFSET, img_size[1]],  # bottom-right corner
        ])
        # Calculate the transformation matrix and it's inverse transformation
        self.M = cv2.getPerspectiveTransform(self.src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, self.src)

        # Define conversions in x and y from pixels space to meters
        self.xm_per_pix = roi_width / (img_size[0] - 2 * DST_OFFSET)
        self.ym_per_pix = (roi_far - roi_near) / img_size[1]

        # Equivalent to cv2.undistort(img, mtx, dist, None, mtx) with the maps computed only once
        self.map1, self.map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, img_size, cv2.CV_16SC2)

    def warp(self, img):
        undist = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(undist, self.M, self.img_size)
        return warped, self.M_inv
