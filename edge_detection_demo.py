import cv2
import pyrealsense2 as rs
from realsense_stream import RS_Stream
from edge_detector import *
import numpy as np

'''
This simply functions as a demo of the first part of the edge detection algorithm.
'''


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
def build_edge_img(src_img: np.ndarray) -> np.ndarray:
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
    return canny_edges


def test1():
    pipe = RS_Stream()

    while(1):
        arr = pipe.get_depth_array()
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(arr, alpha=0.03), cv2.COLORMAP_TWILIGHT)


        cv2.namedWindow('Depth Demo', cv2.WINDOW_NORMAL)
        cv2.imshow('Depth Demo', depth_colormap)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

def test2():
    pipe = RS_Stream()

    while(1):
        arr = pipe.get_rgb_array()
        edge_map = build_edge_img(arr)

        cv2.imshow('output', edge_map)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

test1()
test2()
