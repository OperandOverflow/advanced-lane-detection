### STEP 6: Calculate Vehicle Position and Curve Radius ###

import numpy as np

# Define conversions in x and y from pixels space to meters
YM_PER_PIX = 30 / 720   # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension


def measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty):
    left_fit_cr = np.polyfit(ploty * YM_PER_PIX, left_fitx * XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(ploty * YM_PER_PIX, right_fitx * XM_PER_PIX, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * YM_PER_PIX + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * YM_PER_PIX + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    return left_curverad, right_curverad


def measure_position_meters(binary_warped, left_fit, right_fit):
    # Choose the y value corresponding to the bottom of the image
    y_max = binary_warped.shape[0]
    # Calculate left and right line positions at the bottom of the image
    left_x_pos = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    right_x_pos = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]
    # Calculate the x position of the center of the lane
    center_lanes_x_pos = (left_x_pos + right_x_pos) // 2
    # Calculate the deviation between the center of the lane and the center of the picture
    # The car is assumed to be placed in the center of the picture
    # If the deviation is negative, the car is on the left hand side of the center of the lane
    veh_pos = ((binary_warped.shape[1] // 2) - center_lanes_x_pos) * XM_PER_PIX
    return veh_pos
