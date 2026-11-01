### STEP 8: Lane Finding Pipeline ###

from dataclasses import dataclass

import numpy as np

from .lane_finding import find_lane_pixels_using_histogram, find_lane_pixels_using_prev_poly, fit_poly
from .measurement import measure_curvature_meters, measure_position_meters
from .thresholding import binary_thresholded
from .visualization import project_lane_info

# Number of past polynomial fits averaged to define the search area of the next frame
FIT_HISTORY_LENGTH = 10


@dataclass
class LaneDetection:
    ploty: np.ndarray
    left_fitx: np.ndarray
    right_fitx: np.ndarray
    left_curverad: float
    right_curverad: float
    veh_pos: float
    overlay: np.ndarray


class LaneFindingPipeline:
    """Lane finding on a sequence of BGR frames; keeps the fit history between frames."""

    def __init__(self, perspective):
        self.perspective = perspective
        self.left_fit_hist = []
        self.right_fit_hist = []

    def detect(self, img):
        binary_thresh = binary_thresholded(img)
        binary_warped, M_inv = self.perspective.warp(binary_thresh)

        if not self.left_fit_hist:
            leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)
        else:
            prev_left_fit = np.mean(self.left_fit_hist, axis=0)
            prev_right_fit = np.mean(self.right_fit_hist, axis=0)
            leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_warped, prev_left_fit, prev_right_fit)
            if len(lefty) == 0 or len(righty) == 0:
                leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)

        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped, leftx, lefty, rightx, righty)

        self.left_fit_hist.append(left_fit)
        self.right_fit_hist.append(right_fit)
        if len(self.left_fit_hist) > FIT_HISTORY_LENGTH:
            self.left_fit_hist.pop(0)
            self.right_fit_hist.pop(0)

        xm_per_pix = self.perspective.xm_per_pix
        ym_per_pix = self.perspective.ym_per_pix
        left_curverad, right_curverad = measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty,
                                                                 xm_per_pix, ym_per_pix)
        veh_pos = measure_position_meters(binary_warped, left_fit, right_fit, xm_per_pix)
        overlay = project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv,
                                    left_curverad, right_curverad, veh_pos)

        return LaneDetection(
            ploty=ploty,
            left_fitx=left_fitx,
            right_fitx=right_fitx,
            left_curverad=left_curverad,
            right_curverad=right_curverad,
            veh_pos=veh_pos,
            overlay=overlay,
        )
