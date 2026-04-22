import cv2
import numpy as np

def resize_proportional(img_array, width=None, height=None):
    """
    Resize an image proportionally using either width or height.
    - img_array: input image as a NumPy array (e.g., from cv2.imread)
    - width: desired width (optional)
    - height: desired height (optional)
    Returns: resized image as a NumPy array
    """
    h, w = img_array.shape[:2]

    if width is None and height is None:
        raise ValueError("Either width or height must be specified.")

    if width is not None:
        # Calculate new height to maintain aspect ratio
        aspect_ratio = h / w
        new_height = int(width * aspect_ratio)
        new_size = (width, new_height)
    else:
        # Calculate new width to maintain aspect ratio
        aspect_ratio = w / h
        new_width = int(height * aspect_ratio)
        new_size = (new_width, height)

    resized = cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)
    return resized



