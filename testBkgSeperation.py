# First import the libraries
import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as py # in case we decide to store and export the surface/background data

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

# variables
printedOnce = False
bkgClock = 150 # magic number, until we can figure out a better way
bkground = []


try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        depth_data = depth.as_frame().get_data()

        # depth data is stored as a realsense obj so we need to read it into a numpy array 
        np_image = np.asanyarray(depth_data) 

        # take in and store the background data
        while bkgClock > 0:
            print(f'np_image: q{np_image}')
            bkground.append(np_image)
            # printedOnce = True
            bkgClock = bkgClock-1

        # take the avg of the bkg to smooth over depth data
        mean_bkg = np.mean( np.array(bkground), axis=0 )

        # if not printedOnce:
        #     print(f'bkg0')
        #     for line in bkground[0]:
        #         print(f'{line}\n')
        #     print(f'bkg0')
        #     printedOnce = True

        if not depth:
            continue

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', (np_image-mean_bkg))

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            # print(f'mean bkg')
            # for line in mean_bkg:
            #     print(f'{line}\n')
            # print(f'mean bkg')
            print(f"User pressed break key")
            break

finally:
    pipeline.stop()