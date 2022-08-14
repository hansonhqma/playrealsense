"""
computer vision image manipulation/viewing library for ease of access during development
runs on opencv
"""

import cv2 as cv
import numpy as np

def quickresize(frame:np.ndarray, downscale:float) -> np.ndarray:
    """
    fast image resize

    PARAMETERS

    frame : np.ndarray
        image array

    downscale : float
        image scale multiplier

    RETURNS

    np.ndarray
        resized image
    """
    dim = (int(frame.shape[1]/downscale), int(frame.shape[0]/downscale))
    return cv.resize(frame, dim)

def quickshow(img:np.ndarray, window_name:str = "frame") -> None:
    """
    displays image array in opencv window

    PARAMETERS

    img : np.ndarray
        image array

    window_name : str
        window name
        defaults to "frame"

    RETURNS
    
    None
    """
    cv.imshow(window_name, img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        return

def quickview(source:int, downscale:float=None, label:str="frame") -> None:
    """
    displays video source in opencv window

    PARAMETERS

    downscale : float
        image scale multiplier
        defaults to None

    label : str
        window name
        defaults to "frame"

    RETURNS

    None
    """
    capture = cv.VideoCapture(source)
    while True:
        ret, frame = capture.read()
        if not ret: break

        if downscale:
            frame = quickresize(frame, downscale)
        
        cv.imshow(label, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return
