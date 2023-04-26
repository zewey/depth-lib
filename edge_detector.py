"""
This does much of the heavy lifting for creating the 3 images (diff, edge and blob) necessary for the touch detection
functionality. This is adapted from "DIRECT: Making Touch Tracking on Ordinary Surfaces Practical with Hybrid
Depth-Infrared Sensing," Robert Xiao, Scott Hudson, Chris Harrison. Carnegie Mellon University HCI Institute, 2016.

Relevant links:
--------------
Academic paper: https://robertxiao.ca/pubs/2016_ISS_DIRECT.pdf
DIRECT project github: https://github.com/nneonneo/direct-handtracking

In order to continue work on this project several libraries must be installed:
https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python - python realsense library
https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html - OpenCV for python

2023 - Zooey Schock
"""

import cv2
import numpy as np
from realsense_stream import RS_Stream
from typing import Final
from collections import deque

# -------------- constants for the opencv functions ---------------

GAUSS_KERNEL_SIZE: Final = 5
# standard deviation of gaussian kernel in x and y direction
SIGMA_X: Final = 0
SIGMA_Y: Final = 0
# order of the derivative in the x and y direction
DX: Final = 1
DY: Final = 1
# size of the kernel for sobel
SOBEL_KERNEL_SIZE: Final = 5
CANNY_UPPER_THRESHOLD: Final = 8000
CANNY_LOW_THRESHOLD: Final = 4000
CANNY_APERTURE: Final = 7

# -------------- constants for canny edge pixel categorization ---------------

INSIGNIFICANT: Final = 255
UNVISITED_SIGNIFICANT: Final = 224
SEEN_UNVISITED_SIGNIFICANT: Final = 208
VISITED_SIGNIFICANT: Final = 192
FILL_CANDIDATE: Final = 160
FILLED_SIGNIFICANT: Final = 128
EMPTY_SPACE: Final = 0

# -------------- other constants for edge-mapping functions ---------------

EDGE_REL_DEPTH_DISTANCE: Final = 2
EDGE_REL_DEPTH_THRESHOLD: Final = 50
EDGE_ABS_DEPTH_DISTANCE: Final = 3
EDGE_ABS_DEPTH_THRESHOLD: Final = 100
RED_VALUE_INDEX: Final = 0
GREEN_VALUE_INDEX: Final = 1
BLUE_VALUE_INDEX: Final = 2

# --------------  constants for diff image ---------------

# DIRECT uses 32 bit uints to store the data for the diff img. These corrspond to ARGB, where the alpha channel is used
# to flag something as relevant to display, red channel to indicate the zone, and B+G to indicate the raw difference
#
# our depth image is 24 bits RGB, so we just assign the appropriate value to the R channel
DIFF_IMG_ZONE_ERROR: Final = 0x00       # this is the only zone that will not be assoc. with a flag, bc it is invalid
DIFF_IMG_ZONE_NOISE: Final = 0x00
DIFF_IMG_ZONE_LOW: Final = 0x40
DIFF_IMG_ZONE_MID: Final = 0x80
DIFF_IMG_ZONE_HIGH: Final = 0xC0
# DIRECT uses the high bits in each channel to indicate information so they will be noticeable

# These values are thresholds (in mm) for categorizing which zone in the diff image the pixel will be
DIFF_IMG_ERROR_THRESH: Final = -10
DIFF_IMG_NOISE_THRESH: Final = 0.7
DIFF_IMG_LOW_THRESH: Final = 12
DIFF_IMG_MID_THRESH: Final = 60
# anything above 60 mm is HIGH

FLAGGED: Final = 0xFF       # value to indicate a flagged pixel in the img_flags arrays

# ---------------------------------------------------------


# apply gaussian blur to a given image and return the result
def gauss_blur(src_img):
    img_in = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(img_in, (GAUSS_KERNEL_SIZE, GAUSS_KERNEL_SIZE), SIGMA_X, SIGMA_Y)
    return blurred_image


def display_img(img):
    cv2.imshow('Canny', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------  Edge Map  ---------------

# this function applies algorithms outlined in "Segmentation of Natural Images Using Anisotropic Diffusion and Linking
# of Boundary Edges," by Junji Maeda et al. ('Pattern Recognition," vol. 31 no.12, p 1993-1999. Elsevier, 1998.
#
#
def fill_edge_gaps(canny_img: np.ndarray) -> np.ndarray:
    # *************** FOR TESTING *********************
    # canny_img[canny_img == INSIGNIFICANT] = UNVISITED_SIGNIFICANT
    # *************** FOR TESTING *********************
    w = canny_img.shape[1]    # dimensions of the image stored in ndarray
    h = canny_img.shape[0]    # numpy stores these as (y, x) like it does indexes/coords
    # in DIRECT, irCanny.getPixels() is a method of ofImage on openframeworks, gives reference to start of pixels in
    # memory, so we just pass the ndarray here

    candidate_queue = deque()                           # queue for index of edge candidate pixels

    # test all the neighbors for unvisited, significant pixels
    def find_unvisited_sig(dx: int, dy: int):
        # curr_idx is a tuple in the format (y, x), where (0,0) is the top left - this is how numpy indexes the pixels
        xcoord = curr_idx[1] + dx
        ycoord = curr_idx[0] + dy
        if (0 <= xcoord < w) and (0 <= ycoord < h):
            neighbor_coord = (ycoord, xcoord)         # tuple for coords of neighbor to check
            if canny_img[neighbor_coord] >= SEEN_UNVISITED_SIGNIFICANT:   # we only care about pixels marked >= 208
                nonlocal found
                found += 1
                print('yep')
                if canny_img[neighbor_coord] == UNVISITED_SIGNIFICANT:    # we only queue 224
                    candidate_queue.appendleft(neighbor_coord)
                    canny_img[neighbor_coord] = SEEN_UNVISITED_SIGNIFICANT  # we've tested it now

    # do the same as above function but looking for candidate pixels to be filled in
    def find_fill_cand(dx: int, dy: int):
        xcoord = curr_idx[1] + dx
        ycoord = curr_idx[0] + dy
        if (0 <= xcoord < w) and (0 <= ycoord < h):
            neighbor_coord = (ycoord, xcoord)
            if canny_img[neighbor_coord] == EMPTY_SPACE:       # i.e. if there's nothing here
                candidate_queue.appendleft(neighbor_coord)
                canny_img[neighbor_coord] = FILL_CANDIDATE      # fill the gap

    # indexes will be tuples here so we shouldn't need to do math for the comparisons like they did in DIRECT
    for idx, val in np.ndenumerate(canny_img):
        if canny_img[idx] != UNVISITED_SIGNIFICANT:     # if it's not 224 we don't care about it so skip
            continue

        candidate_queue.appendleft(idx)                 # queue the indexes of pixels that equal 224
        canny_img[idx] = SEEN_UNVISITED_SIGNIFICANT     # update their value as they're seen

        while len(candidate_queue):
            curr_idx = candidate_queue.pop()            # examine all the queued candidate pixels
            curr_pixel = canny_img[curr_idx]
            if curr_pixel == SEEN_UNVISITED_SIGNIFICANT:
                canny_img[curr_idx] = VISITED_SIGNIFICANT   # update as this is now visited

            found = 0
            find_unvisited_sig(-1, -1); find_unvisited_sig(-1, 0); find_unvisited_sig(-1, 1); find_unvisited_sig(0,-1); find_unvisited_sig(0, 1)
            find_unvisited_sig(1, -1); find_unvisited_sig(1, 0); find_unvisited_sig(1, 1)

            if curr_pixel == FILL_CANDIDATE:
                if found:
                    canny_img[curr_idx] = FILLED_SIGNIFICANT
                else:
                    canny_img[curr_idx] = EMPTY_SPACE
            elif not found:
                # mark all neighbors as fill candidates
                find_fill_cand(-1, -1); find_fill_cand(-1, 0); find_fill_cand(-1, 1); find_fill_cand(0, -1); find_fill_cand(0, 1)
                find_fill_cand(1, -1); find_fill_cand(1, 0); find_fill_cand(1, 1)

    return canny_img


# Build the edge map using Canny edge detection algorithm ("A Computational Approach to Edge Detection," John Canny.
# 'IEEE Transactions on Pattern Analysis and Machine Intelligence' Vol. PAMI-8 no.6. IEEE, 1986)
def build_edge_img(src_img: np.ndarray, image_flags: np.ndarray, depth_image: np.ndarray) -> (np.ndarray, np.ndarray):
    w = src_img.shape[1]    # dimensions of the image stored in ndarray
    h = src_img.shape[0]    # numpy stores these as (y, x) like it does indexes/coords

    # get an ndarray for the return value, same size as src image, but with 3 channels at each x,y for RGB
    edge_pixels = np.zeros(shape=(w, h, 3), dtype=np.uint8)    # get an ndarray of the same size, but zero'd, aka edgePx in DIRECT

    # convert the input img to greyscale, apply gaussian blur filter, to help with canny
    greyscale_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    gauss_image = cv2.GaussianBlur(greyscale_img, (GAUSS_KERNEL_SIZE, GAUSS_KERNEL_SIZE), SIGMA_X, SIGMA_Y)
    # aka ircannyPx in DIRECT
    canny_edges = cv2.Canny(image=gauss_image, threshold1=CANNY_LOW_THRESHOLD, threshold2=CANNY_UPPER_THRESHOLD, apertureSize=CANNY_APERTURE, L2gradient=True)

    # ~~ DIRECT uses the 8-bit greyscale pixel values to categorize the pixels ~~ #
    # to start, all pixels (that aren't black indicating no edge detected by canny) are marked 224, indicating the pixel
    # is significant and has not had its neighbors examined yet
    canny_edges[canny_edges == INSIGNIFICANT] = UNVISITED_SIGNIFICANT
    canny_edges = fill_edge_gaps(canny_edges)

    # build the final edge map

    for idx, val in np.ndenumerate(canny_edges):
        if canny_edges[idx]:
            # in DIRECT, image pixels are 32 bit ARGB, where A is used (in this image and all others) as a flag to indicate
            # that there is information in this cell. RGB are used as they are normally. Our colors are 24 bit (3 x 8bit)
            # so we need image_flags to do this job instead, as a parallel array
            edge_pixels[idx][0] = val
            image_flags[idx] = FLAGGED
        continue

    # for the d435, the depth and color streams are the same size - if a different model of camera were used, and this
    # was not the case, this will break
    try:
        assert src_img.shape[0] == image_flags.shape[0] == depth_image.shape[0]
        assert src_img.shape[1] == image_flags.shape[1] == depth_image.shape[1]

        # all the ndarrays have the same w & h, so we will use the simplest array for it's index
        for idx, val in np.ndenumerate(image_flags):
            if idx[0] < EDGE_REL_DEPTH_DISTANCE:    # DIRECT ignores some of the pixels close to the edge
                continue
            curr_val = 0
            curr_val |= (depth_image[idx][1] << 8) | depth_image[idx][2]   # we want to look at the G B channels, just ram these together to make the comparison easier
            for dx in range(-1,2):
                for dy in range(-1,2):
                    neighbor = 0
                    # do the same as we did above for the current pixel, for comparison
                    neighbor |= (depth_image[idx[0] + dy][idx[1] + dx][1] << 8) | depth_image[idx[0] + dy][idx[1] + dx][2]
                    # DIRECT excludes pixels that differ greatly from, their neighbors. 'Exclude' means that they are
                    # excluded from the blobimage; If the difference between neighbors is so great, then this pixel
                    # is an edge, and belongs in the edge map, not the blob map
                    if abs(curr_val - neighbor) > EDGE_REL_DEPTH_THRESHOLD:
                        # add to edge map
                        # the value being entered in the G value indicates it is a depth-relative edge, i.e. that difference
                        # between this pixel and neighbor is large
                        edge_pixels[idx][GREEN_VALUE_INDEX] |= 0xFF
                        image_flags[idx] |= FLAGGED
                        break
                else:               # this is the easiest way I could find to break out of the 2 inner loops
                    continue
                break
        # this is like the above set of loops, but dealing with large differences between the current pixel and the background
        for idx, val in np.ndenumerate(image_flags):
            if idx[0] < EDGE_ABS_DEPTH_DISTANCE:
                continue
            for dx in range(-1,2):
                for dy in range(-1,2):
                    difference_value = 0
                    difference_value |= (depth_image[idx[0] + dy][idx[1] + dx][1] << 8) | depth_image[idx[0] + dy][idx[1] + dx][2]
                    # this will reject (add to edge map and exclude from blob image) if the other pixel is too high above background
                    if difference_value > EDGE_ABS_DEPTH_THRESHOLD:
                        edge_pixels[idx][BLUE_VALUE_INDEX] |= 0xFF
                        image_flags[idx] |= FLAGGED
                        break
                else:
                    continue
                break
    except AssertionError:
        # if we're here something has gone horribly awry
        print('Error. X/Y dimensions of input images are not equal.\n')

    return (edge_pixels, image_flags)

# --------------  Difference Image  ---------------


# this will update the difference image using the raw depth image provided by the camera
#
# diff_img will be all zeros the first time this runs
def build_diff_img(new_depth_frame: np.ndarray, diff_img: np.ndarray, image_flags: np.ndarray) -> (np.ndarray, np.ndarray):
    # dimensions, same as build_edge_img
    w = new_depth_frame.shape[1]
    h = new_depth_frame.shape[0]

    # we'll need the mean and std. deviation to construct the diff img
    # in an ideal world, the background detection function runs at around 15 fps in a separate thread or process,
    # and utilizes a rolling window of ~ 10 data points per pixel. In this way, each pixel will have its own mean
    # and standard deviation which will be used in measuring the difference between the background and whatever is
    # occupying that space currently.
    #
    # Background pixels will not be updated if the height at that x,y position is above a threshold, to prevent anything
    # other than the background from becoming part of the background.
    bg_mean = 0.           # these will need to be populated by a function returning an array for this, one of each
    bg_std_dev = 0.        # for each coord of the depth img

    # check this, like in the edge map function
    assert new_depth_frame.shape[0] == image_flags.shape[0] == diff_img.shape[0]
    assert new_depth_frame.shape[1] == image_flags.shape[1] == diff_img.shape[1]
    # update the diff image
    for idx, val in np.ndenumerate(image_flags):
        diff = 0.
        z_score = 0.
        # realsense seems to store depth information in a 16 bit uint, lower number being closer to the lens
        # distance is in mm
        if new_depth_frame[idx]:
            diff = bg_mean[idx] - new_depth_frame[idx]
            z_score = diff / bg_std_dev[idx]
        else:
            diff = 0
            z_score = 0

        # Once the ability to retrieve the per-pixel std. dev. and mean is available, this section builds out the
        # diff image
        #
        # The flag image is marked where there is relevant data to display, like the edge map section. As per correspondence
        # with Dr. Xiao, the red channel is used to visually categorize the depth information into 3 zones (low, mid, high)
        # while the blue and green channels store raw difference values
        if (bg_mean[idx] == 0) or (diff < DIFF_IMG_ERROR_THRESH):
            image_flags[idx] = 0
            diff_img[idx][RED_VALUE_INDEX] = DIFF_IMG_ZONE_ERROR
            diff_img[idx][GREEN_VALUE_INDEX] = abs(diff >> 8)
            diff_img[idx][BLUE_VALUE_INDEX] = abs(diff & 0x00FF)
        elif z_score < DIFF_IMG_NOISE_THRESH:
            image_flags[idx] = FLAGGED
            diff_img[idx][RED_VALUE_INDEX] = DIFF_IMG_ZONE_NOISE
            diff_img[idx][GREEN_VALUE_INDEX] = diff >> 8                # isolate the high bits from the 16 bit depth val
            diff_img[idx][BLUE_VALUE_INDEX] = (diff & 0x00FF)           # isolate the low bits from the 16 bit depth val
        elif diff < DIFF_IMG_LOW_THRESH:
            image_flags[idx] = FLAGGED
            diff_img[idx][RED_VALUE_INDEX] = DIFF_IMG_ZONE_LOW
            diff_img[idx][GREEN_VALUE_INDEX] = diff >> 8
            diff_img[idx][BLUE_VALUE_INDEX] = (diff & 0x00FF)
        elif diff < DIFF_IMG_MID_THRESH:
            image_flags[idx] = FLAGGED
            diff_img[idx][RED_VALUE_INDEX] = DIFF_IMG_ZONE_MID
            diff_img[idx][GREEN_VALUE_INDEX] = diff >> 8
            diff_img[idx][BLUE_VALUE_INDEX] = (diff & 0x00FF)
        else:
            image_flags[idx] = FLAGGED
            diff_img[idx][RED_VALUE_INDEX] = DIFF_IMG_ZONE_HIGH
            diff_img[idx][GREEN_VALUE_INDEX] = diff >> 8
            diff_img[idx][BLUE_VALUE_INDEX] = (diff & 0x00FF)

    return (diff_img, image_flags)

# --------------  Flood Filling  ---------------


def flood_arm(diff_img: np.ndarray, blob_img: np.ndarray, blobID: int) -> ():
    