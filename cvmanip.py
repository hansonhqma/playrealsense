import cv2 as cv
import numpy as np


def quickresize(frame:np.ndarray, downscale:int) -> np.ndarray:
    dim = (int(frame.shape[1]/downscale), int(frame.shape[0]/downscale))
    return cv.resize(frame, dim)

def quickshow(img:np.ndarray, window_name:str = "frame") -> None:
    cv.imshow(window_name, img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        return

def quickview(camera_index:int, downscale:int=None, label:str="frame") -> None:
    capture = cv.VideoCapture(camera_index)
    while True:
        ret, frame = capture.read()
        if not ret: break

        if downscale:
            frame = quickresize(frame, downscale)
        
        cv.imshow(label, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            return

