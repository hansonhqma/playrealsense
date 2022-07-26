import cv2 as cv
import cvmanip as cvm
import pyrealsense2 as rs

import numpy as np
import time
import sys

from collections import deque

def sa_ratio(contour:np.ndarray) -> float:
    # TODO: wip

    area = cv.contourArea(contour)
    return contour.shape[0]

def depth_fusion(phi:float, theta:float, depth:float) -> tuple:
    """
    calculates relative cartesian coordinate of target

    PARAMETERS:
    
    phi : float
        angle from y axis (left +) in radians

    theta : float
        angle from x axis (up +) in radians

    depth : float
        depth value of target
    """

    x = depth*np.sin(phi)
    y = depth*np.sin(theta)

    zy_plane_hyp = np.sqrt(np.power(depth, 2) - np.power(x, 2))
    z = np.sqrt(np.power(zy_plane_hyp, 2) - np.power(y, 2))

    return (x,y,z)

def sample_depth(x_pos:float, y_pos:float, simulate:bool=True) -> float:
    """
    function for sampling depth information from realsense unit

    PARAMETERS:

    x_pos : float
        x position of target relative to fov
        e.g right edge of frame is +1, left edge is -1

    y_pos : float
        y position of target relative to fov
        e.g top edge of frame is +1, bottom edge is -1

    simulate : bool
        enables or disables simulation mode
        true for dummy depth value
        false for actual depth value from realsense unit, defaults to true
    """

record = '-r' in sys.argv


bit_shift = 5
downscale = 1

hsv_min = (0,0,0)
hsv_max = (180,255,55)

FRAMERATELOG = deque(maxlen=1000)

# metadata for D415 realsense
focal_length = 1.88 # mm
sensor_width = 1.3018 # mm
sensor_height = 0.7311 # mm

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
simulate = False
try:
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
except:
    print("No camera connected - in simulation mode")
    simulate = True

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config) # start pipeline

width = 640
height = 480
frame_center = (width//2, height//2)

"""
# COLOR VIDEO CAPTURE
source = "test media/bundle.mov"
capture = cv.VideoCapture(source)
ret, frame = capture.read()
if not ret:
    exit()

width, height = int(frame.shape[1]//downscale), int(frame.shape[0]//downscale)

frame_center = (width//2, height//2)

"""

if record:
    writer = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

d = 2

while True:
    start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)

    #out, frame = capture.read()
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        break

    #frame = cvm.quickresize(frame, downscale)
    depth_image = np.asanyarray(depth_frame.get_data())
    
    frame = np.asanyarray(color_frame.get_data())
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = frame.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        images = np.hstack((resized_frame, depth_colormap))
    else:
        images = np.hstack((frame, depth_colormap))


    flattened = (frame >> bit_shift) << bit_shift # flatten to 8 colors

    binary_frame = cv.inRange(cv.cvtColor(flattened, cv.COLOR_BGR2HSV), hsv_min, hsv_max)

    #edges = cv.Canny(flattened, 200, 300)
    contours, hierarchy= cv.findContours(binary_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # assumption: binary image produces closed contours
    # -> we can find volume

    if len(contours) > 0:
        # a "cable shape" can be defined as an object with high size to area ratio
        target_contour = max(contours, key=sa_ratio)
    
        moment = cv.moments(target_contour)

        target_center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))

        # TODO: make this a function
        rel_x = (sensor_width/2)*((width/2-target_center[0])/(width/2))
        rel_y = (sensor_height/2)*((height/2-target_center[1])/(height/2))

        angle_from_y_axis = np.arctan(rel_x/focal_length)
        angle_from_x_axis = np.arctan(rel_y/focal_length)

        frame = cv.drawContours(frame, [target_contour], 0, (0,0,255), 3)

        #frame = cv.circle(frame, target_center, 20, (255,255,0), 1)
        #angle = (np.rad2deg(angle_from_y_axis), np.rad2deg(angle_from_x_axis))
        #frame = cv.putText(frame, "{:.2f}, {:.2f}".format(angle[0], angle[1]), target_center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        d = depth_frame.get_distance(target_center[0], target_center[1])

        cc = depth_fusion(angle_from_y_axis, angle_from_x_axis, d)
        frame = cv.line(frame, frame_center, target_center, (0,255,255), 2)
        frame = cv.putText(frame, "x:{:.3f}, y:{:.3f}, z:{:.3f}".format(cc[0], cc[1], cc[2]), frame_center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 255), 1)

        
    if record:
        writer.write(frame)

    cv.imshow("frame", frame)
    cv.imshow("depth", images)
    #cv.imshow("edges", binary_frame)

    time_delta = (time.clock_gettime_ns(time.CLOCK_REALTIME)-start_time)/1e9
    FRAMERATELOG.append(1/time_delta)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

if record:
    writer.release()

print("Average framerate = {:2f}".format(np.sum(FRAMERATELOG)/len(FRAMERATELOG)))
