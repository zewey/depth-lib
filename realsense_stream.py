

import pyrealsense2 as rs2
import cv2
import numpy as np


class RS_Stream:
    def __init__(self):
        self.pipe = rs2.pipeline()
        config = rs2.config()

        # The following is script common to all example of Realsense use-cases, for determining specific models,
        # and configuring accordingly
        #
        # src: https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs2.pipeline_wrapper(self.pipe)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs2.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs2.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs2.stream.depth, 640, 480, rs2.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs2.stream.color, 960, 540, rs2.format.bgr8, 30)
        else:
            config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 30)

        # Start streaming
        self.pipe.start(config)

        # End of model determination-configuration #

    # return an rgb image (as numpy ndarray)
    def get_rgb_array(self) -> np.ndarray:
        frame_set = self.pipe.wait_for_frames()
        rgb_frame = frame_set.get_color_frame()
        return np.asanyarray(rgb_frame.get_data())

    # return a depth image (as numpy ndarray)
    def get_depth_array(self) -> np.ndarray:
        frame_set = self.pipe.wait_for_frames()
        depth_frame = frame_set.get_depth_frame()
        depth_img = np.asanyarray(depth_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_COOL)

        return depth_img

    # not sure if calling the above 2 functions, each with their own wait_for_frames() will desync
    def get_all_arrays(self) -> (np.ndarray, np.ndarray):
        depth_frame = self.get_depth_array()
        rgb_frame = self.get_rgb_array()
        return rgb_frame, depth_frame

    # kill the camera stream
    def stop_stream(self):
        self.pipe.stop()

#pipe = RS_Stream()
