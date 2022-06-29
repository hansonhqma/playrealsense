import cv2 as cv
import numpy as np
import rapidcv
import time
import sys

from collections import deque

record = '-r' in sys.argv

capture = cv.VideoCapture("bundle.mov")

bit_shift = 5
downscale = 1.5

FRAMERATELOG = deque(maxlen=1000)

ret, frame = capture.read()
if not ret:
    exit()

width, height = int(frame.shape[1]//downscale), int(frame.shape[0]//downscale)

if record:
    writer = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

while True:
    start_time = time.clock_gettime_ns(time.CLOCK_REALTIME)

    out, frame = capture.read()
    if not out:
        break

    frame = cv.resize(frame, (int(frame.shape[1]//downscale), int(frame.shape[0]//downscale)))

    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    flattened = (grayscale >> bit_shift) << bit_shift # flatten to 8 colors

    edges = cv.Canny(flattened, 250, 300)
    contours, hierarchy= cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=lambda x : x.shape[0])

    frame = cv.drawContours(frame, [largest_contour], 0, (0,255,0), 4)

    if record:
        writer.write(frame)

    cv.imshow("frame", frame)
    cv.imshow("edges", edges)

    time_delta = (time.clock_gettime_ns(time.CLOCK_REALTIME)-start_time)/1e9
    FRAMERATELOG.append(1/time_delta)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


writer.release()
print("Average framerate = {:2f}".format(np.sum(FRAMERATELOG)/len(FRAMERATELOG)))