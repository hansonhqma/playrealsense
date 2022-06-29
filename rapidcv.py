import cv2 as cv
import numpy as np

def quickshow(img:np.ndarray, window_name:str = "frame") -> None:
    cv.imshow(window_name, img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        return