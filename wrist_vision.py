import cv2 as cv
import cvmanip as cvm
import pyrealsense2 as rs

import numpy as np
import time
import sys
import json

import socket #TODO: stream position info to localhost

from collections import deque

def depth_fusion(phi:float, theta:float, depth:float) -> tuple:
    """
    calculates relative cartesian coordinate of target

    PARAMETERS
    
    phi : float
        angle from y axis (left +) in radians

    theta : float
        angle from x axis (up +) in radians

    depth : float
        depth value of target

    RETURNS

    tuple
        x, y, z position relative to camera reference frame
    """

    x = depth*np.sin(phi)
    y = depth*np.sin(theta)

    zy_plane_hyp = np.sqrt(np.power(depth, 2) - np.power(x, 2))
    z = np.sqrt(np.power(zy_plane_hyp, 2) - np.power(y, 2))

    return (x,y,z)

def contour_centroid(contour:np.ndarray) -> tuple:

    moment = cv.moments(contour)
    if moment["m00"] == 0:
        return None
    return (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))

def pixel_angles(target_pixel:tuple, frame_dim:tuple, sensor_dim:tuple, focal_length:float) -> tuple:
    """
    calculates azimuth and altitude angles of a given pixel location relative to camera reference frame

    PARAMETERS

    target_pixel : tuple
        pixel coordinate in width, height
    
    frame_dim : tuple
        frame dimensions in width, height
    
    sensor_dim : tuple
        sensor dimensions in width, height, in mm
    
    focal_length : float
        focal length of lens, in mm

    RETURNS

    tuple
        azimuth, altitude angles, respectively
    """
    rel_x = (sensor_dim[0]/2)*((frame_dim[0]/2-target_pixel[0])/(frame_dim[0]/2))
    rel_y = (sensor_dim[1]/2)*((frame_dim[1]/2-target_pixel[1])/(frame_dim[1]/2))

    return np.arctan(rel_x/focal_length), np.arctan(rel_y/focal_length)
    

# set modes
record = '-r' in sys.argv
debug = '-d' in sys.argv

# ingest camera and runtime info from JSON files
CAMERA_INFO = json.load(open("d415.json"))
RUNTIME_INFO = json.load(open("runtime.json"))

# ingest color sensor data
COLOR_SENSOR_DATA = CAMERA_INFO["color_sensor_data"]
focal_length = COLOR_SENSOR_DATA["focal_length"]
sensor_width = COLOR_SENSOR_DATA["sensor_width"]
sensor_height = COLOR_SENSOR_DATA["sensor_height"]

# ingest runtime data
STREAM_INFO = RUNTIME_INFO["stream_info"]
HSV_DATA = RUNTIME_INFO["hsv_data"]
stream_width = int(STREAM_INFO['frame_width'])
stream_height = int(STREAM_INFO['frame_height'])
stream_framerate = int(STREAM_INFO['framerate'])
bit_shift = int(RUNTIME_INFO["colorspace_flatten_shamt"])

# init framerate log container
FRAMERATELOG = deque(maxlen=1000)

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

config.enable_stream(rs.stream.depth, stream_width, stream_height, rs.format.z16, stream_framerate)
config.enable_stream(rs.stream.color, stream_width, stream_height, rs.format.bgr8, stream_framerate)
pipeline.start(config) # start pipeline

frame_center = (stream_width//2, stream_height//2)

if record:
    writer = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*"mp4v"), stream_framerate, (stream_width, stream_height))

while True:
    # start clock for fps
    start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)

    # get image data from realsense unit
    frames = pipeline.wait_for_frames()
    depth_data = frames.get_depth_frame()
    color_data = frames.get_color_frame()

    if not depth_data or not color_data: # fails to get frame
        break

    # convert image data to image array
    depth_frame = np.asanyarray(depth_data.get_data())
    color_frame = np.asanyarray(color_data.get_data())


    # color space flatten
    flattened = (color_frame >> bit_shift) << bit_shift

    # hsv threshold
    binary_frame = cv.inRange(cv.cvtColor(flattened, cv.COLOR_BGR2HSV), hsv_min, hsv_max)

    # contour search
    contours, hierarchy= cv.findContours(binary_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if not len(contours) == 0: # if there are contours in the image

        # find largest contour
        target_contour = max(contours, key=lambda x:x.shape[0]) 
    
        # calculate grasp point
        grasp_point = contour_centroid(target_contour)

        # calculate azimuth and altitude angles of grasp point
        
        azimuth, altitude = pixel_angles(grasp_point, )

        # TODO: make this a function

        color_frame = cv.drawContours(color_frame, [target_contour], 0, (0,0,255), 3)

        d = depth_data.get_distance(target_center[0], target_center[1])

        cc = depth_fusion(angle_from_y_axis, angle_from_x_axis, d)
        color_frame = cv.line(color_frame, frame_center, target_center, (0,255,255), 2)
        color_frame = cv.putText(color_frame, "x:{:.3f}, y:{:.3f}, z:{:.3f}".format(cc[0], cc[1], cc[2]), frame_center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 255), 1)

        
    if record:
        writer.write(color_frame)

    cv.imshow("frame", color_frame)

    time_delta = (time.clock_gettime_ns(time.CLOCK_REALTIME)-start_time)/1e9
    FRAMERATELOG.append(1/time_delta)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

if record:
    writer.release()

print("Average framerate = {:2f}".format(np.sum(FRAMERATELOG)/len(FRAMERATELOG)))
