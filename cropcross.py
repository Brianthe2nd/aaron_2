"""This script will be used to find the location of a cross using cv2 template matching then 
it will return the x coordinate of the first cross in the x axis """


import cv2
import numpy as np
from logo import detect_best_logo_height
from resize import resize_proportional

def cross_add(logo_size):
    # 44 = 29
    # logo_size
    """NOTE: This logo size should be the logosize that
    has been measured at the bottom if statement"""
    return round((logo_size*29) / 52)

def cross_size(logo_size):


    return round((logo_size * 21)/ 52) 

def deduplicate_points(points, min_dist=20):
    filtered = []
    for p in points:
        if all(abs(p[0] - q[0]) > min_dist for q in filtered):
            filtered.append(p)
    return filtered



def get_cross(image, logo_loc, logo_height):
    top_left_roi, bottom_right_roi = logo_loc
    h, w = image.shape[:2]

    # Crop ROI vertically, keep full width
    cropped = image[top_left_roi[1]:bottom_right_roi[1] + 20, :]

    # Load and resize template
    cross_template = cv2.imread("templates/cross.png")
    ideal_height = cross_size(logo_height)
    cross_template = resize_proportional(cross_template, ideal_height)

    th, tw = cross_template.shape[:2]

    # Template matching
    res = cv2.matchTemplate(cropped, cross_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
    ys, xs = np.where(res >= threshold)

    if len(xs) == 0:
        return None, None

    # Deduplicate matches
    points = list(zip(xs, ys))
    points = deduplicate_points(points, min_dist=tw // 2)

    left_crop = None
    right_crop = None

    # Loop through all detected crosses
    for x, y in points:
        abs_x = x  # full-width crop, so x is already absolute

        if abs_x < w / 2:
            # Cross on left → crop left side
            candidate = abs_x + tw
            if left_crop is None or candidate > left_crop:
                left_crop = candidate
        else:
            # Cross on right → crop right side
            candidate = abs_x + tw
            if right_crop is None or candidate < right_crop:
                right_crop = candidate

    return left_crop, right_crop


def crop_right(image , logo_loc , logo_height):
    corrected = None
    leftcrop,right_crop = get_cross(image ,logo_loc,logo_height)
    crossAdd = cross_add(logo_height)
    if right_crop:
        h,w = image.shape[:2]
        
        right_b = min(w,right_crop + crossAdd)
        corrected = image[:, 0:right_b] 
    else:
        corrected = image.copy()
    if leftcrop:
        corrected = image[:,leftcrop+crossAdd:]
    
    if corrected is None:
        corrected = image.copy()
    return corrected
    

if __name__ == "__main__":
    image = cv2.imread("frame.png")
    h,w = image.shape[:2]
    logo_height , logo_sim , logo_loc = detect_best_logo_height(image)
    print("logo loc")
    print(logo_loc)
    print(logo_height)
    from dump import display_image
    image = crop_right(image,logo_loc[0],logo_height)
    display_image(image)
    # x = get_cross(image)
    # cv2.line(image,(x,0),(x,h),(0,255,0),1)
    # cv2.imshow(image)
    print("The logo height is: ",logo_height)
    
