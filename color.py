import cv2
import numpy as np
# from main import Print
from std_out import Print,play_error_sound,log_exception
import math

def is_hue_in_range(h, s, v,expected):
    """
    Input:
        h: 0–179 (OpenCV hue)
        s: 0–255
        v: 0–255
    
    Output:
        "red", "green", "gray", or "other"
    """

    # ---- 1. Convert HSV → RGB (OpenCV expects values as uint8) ----
    hsv_pixel = np.uint8([[[h, s, v]]])
    rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0,0]

    # ---- 2. Convert RGB → Lab ----
    lab = cv2.cvtColor(np.uint8([[rgb_pixel]]), cv2.COLOR_RGB2LAB)[0,0]
    L, a, b = lab

    # ---- 3. Remove OpenCV offset ----
    a = int(a) - 128
    b = int(b) - 128


    # ---- 4. Lab chroma ----
    chroma = math.sqrt(a*a + b*b)

    # ---- 5. Gray threshold ----
    # Good threshold = 12 (from Lab perceptual studies)
    if chroma < 12:
        return False

    hue = h * 2

    if 81 <= hue <= 140 and expected == "green":
        return True

    if (hue <= 15 or hue >= 330) and expected == "red":
        return True

    return False


import cv2
import numpy as np

def red_percentage(image):
    """
    Returns the percentage of the image that is perceptually red
    using Lab chroma + hue logic (vectorized, no loops).
    """

    # ---- 1. BGR → HSV ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1]
    v = hsv[..., 2]

    # ---- 2. HSV → RGB (vectorized) ----
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # ---- 3. RGB → Lab (vectorized) ----
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # ---- 4. Remove OpenCV Lab offset ----
    a = lab[..., 1].astype(np.int16) - 128
    b = lab[..., 2].astype(np.int16) - 128

    # ---- 5. Compute chroma ----
    chroma = np.sqrt(a * a + b * b)

    # ---- 6. Gray rejection ----
    not_gray = chroma >= 12

    # ---- 7. Hue logic (convert OpenCV hue → degrees) ----
    hue_deg = h * 2
    red_hue = (hue_deg <= 15) | (hue_deg >= 330)

    # ---- 8. Final red mask ----
    red_mask = not_gray & red_hue

    # ---- 9. Percentage ----
    percent_red = red_mask.mean() * 100
    return percent_red


def green_percentage(image):
    """
    Returns the percentage of the image that is perceptually green
    using Lab chroma + hue logic (vectorized, no loops).
    """

    # ---- 1. BGR → HSV ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1]
    v = hsv[..., 2]

    # ---- 2. HSV → RGB (vectorized) ----
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # ---- 3. RGB → Lab (vectorized) ----
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # ---- 4. Remove OpenCV Lab offset ----
    a = lab[..., 1].astype(np.int16) - 128
    b = lab[..., 2].astype(np.int16) - 128

    # ---- 5. Compute chroma ----
    chroma = np.sqrt(a * a + b * b)

    # ---- 6. Gray rejection ----
    not_gray = chroma >= 12

    # ---- 7. Hue logic (convert OpenCV hue → degrees) ----
    hue_deg = h * 2
    green_hue = (81 <= hue_deg) & (hue_deg <= 140)

    # ---- 8. Final green mask ----
    green_mask = not_gray & green_hue

    # ---- 9. Percentage ----
    percent_green = green_mask.mean() * 100
    return percent_green




def orange_percentage(image, lower_orange=(10, 100, 100), upper_orange=(25, 255, 255)):
    """
    Returns the percentage of the image that is orange.

    Parameters:
        image (np.array): BGR image (as read by cv2).
        lower_orange, upper_orange: Lower and upper HSV bounds for orange color.

    Returns:
        float: Percentage of orange pixels in the image (0 to 100).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask for orange color
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    orange_pixels = cv2.countNonZero(orange_mask)
    total_pixels = image.shape[0] * image.shape[1]

    percent_orange = (orange_pixels / total_pixels) * 100
    return percent_orange


def gray_percentage(image, lower_gray = (0, 0, 0), upper_gray=(180, 50, 220)):
    """
    Returns the percentage of the image that is gray based on HSV value range.
    
    Parameters:
        image (np.array): BGR image.
        lower_gray, upper_gray: HSV bounds to define gray color.

    Returns:
        float: Percentage of gray pixels.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_mask = cv2.inRange(hsv, np.array(lower_gray), np.array(upper_gray))
    gray_pixels = cv2.countNonZero(gray_mask)
    total_pixels = image.shape[0] * image.shape[1]
    
    return ((gray_pixels / total_pixels) * 100) + 15

# def green_percentage(image, lower_green=(35, 40, 40), upper_green=(85, 255, 255)):
#     """
#     Returns the percentage of the image that is green.

#     Parameters:
#         image (np.array): BGR image.
#         lower_green, upper_green: HSV bounds for green.

#     Returns:
#         float: Percentage of green pixels.
#     """
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     green_mask = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green))
#     green_pixels = cv2.countNonZero(green_mask)
#     total_pixels = image.shape[0] * image.shape[1]
#     return (green_pixels / total_pixels) * 100

def blue_percentage(image, lower_blue=(100, 150, 50), upper_blue=(130, 255, 255)):
    """
    Returns the percentage of the image that is blue.

    Parameters:
        image (np.array): BGR image (as read by cv2).
        lower_blue, upper_blue: Lower and upper HSV bounds for blue color.

    Returns:
        float: Percentage of blue pixels in the image (0 to 100).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask for blue color
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_pixels = cv2.countNonZero(blue_mask)
    total_pixels = image.shape[0] * image.shape[1]

    percent_blue = (blue_pixels / total_pixels) * 100
    return percent_blue


def process_color(image):
    """
    Processes the image to calculate the percentage of red, gray, and green colors.

    Parameters:
        image (np.array): BGR image.

    Returns:
        dict: Dictionary with percentages of red, gray, and green.
    """
    return {
        "red": red_percentage(image),
        "gray": gray_percentage(image),
        "green": green_percentage(image)
    }
    
import cv2
import numpy as np
from collections import Counter
from PIL import Image
import os


    
import cv2
import numpy as np
from collections import Counter
from PIL import Image
import os

def is_gray_hsv(h, s, v, sat_threshold=60, min_value=150):
    """
    Determines whether an HSV color should be classified as gray,
    including very light gray / near-white colors.
    
    Parameters:
        h, s, v : int
            Hue, Saturation, and Value components (H:0–180, S/V:0–255)
        sat_threshold : int
            Maximum S allowed to still be considered gray.
        min_value : int
            Minimum V required (prevents dark colors from being classified as white/gray).
    
    Returns:
        bool : True if the color is gray (including near-white), False otherwise.
    """
    # Must be low saturation AND sufficiently bright (near white/light gray)
    return s <= sat_threshold and v >= min_value


def _most_common_color_from_array(section_rgb, downsample_factor=1):
    """Find the most common RGB color in a numpy RGB array (HxWx3)."""
    if downsample_factor > 1:
        section_rgb = section_rgb[::downsample_factor, ::downsample_factor]

    pixels = section_rgb.reshape(-1, 3)
    tuples = [tuple(p) for p in pixels.astype(np.uint8)]
    counter = Counter(tuples)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else (0, 0, 0)


def _black_white_percentage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu splits image into two clusters: dark cluster and bright cluster
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Two groups:
    dark_cluster  = gray[binary == 0]
    bright_cluster = gray[binary == 255]
    # cv2.imshow("binary",bright_cluster)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # The smaller region is usually the TEXT (thin strokes)
    if len(dark_cluster) < len(bright_cluster):
        text_group = dark_cluster
    else:
        text_group = bright_cluster

    # Check if text pixels are bright or dark
    if np.mean(text_group) > 128:
        return "white"
    else:
        return "black"



def _hsv_from_rgb(rgb):
    """Convert an RGB tuple to HSV (H:0–360, S:0–255, V:0–255)."""
    # Convert RGB → BGR → HSV
    bgr = np.uint8([[rgb[::-1]]])  # RGB → BGR
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    h = int(hsv[0, 0, 0])   # convert OpenCV hue 0–180 → 0–360
    s = int(hsv[0, 0, 1])
    v = int(hsv[0, 0, 2])

    return h, s, v

import cv2
import numpy as np

def count_grey_pixels(image_bgr, chroma_threshold=5):
    """
    image_bgr: OpenCV BGR image
    chroma_threshold: max difference allowed between channels
                      (0 means perfect grey, usually 5–15 is ok in real images)

    Returns: number of grey-ish pixels
    """

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Split channels
    R = image_rgb[:, :, 0].astype(np.int16)
    G = image_rgb[:, :, 1].astype(np.int16)
    B = image_rgb[:, :, 2].astype(np.int16)

    # Compute chroma = max - min per pixel
    max_c = np.maximum(np.maximum(R, G), B)
    min_c = np.minimum(np.minimum(R, G), B)
    chroma = max_c - min_c

    # Grey mask
    grey_mask = chroma <= chroma_threshold

    # Count grey pixels
    return int(np.sum(grey_mask))

def count_blue_hue_pixels(image_bgr):
    """
    image_bgr: np.ndarray (OpenCV-loaded image in BGR format)
    Returns: number of pixels with 81 <= H <= 140
    """

    # Convert BGR → HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Extract hue channel
    h_channel = hsv[:, :, 0]

    # Create mask for hue range
    mask = (h_channel >= 81) & (h_channel <= 140)

    # Count pixels that match
    return int(np.sum(mask))


def is_hue_in_range(h, s, v,expected):
    """
    Input:
        h: 0–179 (OpenCV hue)
        s: 0–255
        v: 0–255
    
    Output:
        "red", "green", "gray", or "other"
    """

    # ---- 1. Convert HSV → RGB (OpenCV expects values as uint8) ----
    hsv_pixel = np.uint8([[[h, s, v]]])
    rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0,0]

    # ---- 2. Convert RGB → Lab ----
    lab = cv2.cvtColor(np.uint8([[rgb_pixel]]), cv2.COLOR_RGB2LAB)[0,0]
    L, a, b = lab

    # ---- 3. Remove OpenCV offset ----
    a = int(a) - 128
    b = int(b) - 128


    # ---- 4. Lab chroma ----
    chroma = math.sqrt(a*a + b*b)

    # ---- 5. Gray threshold ----
    # Good threshold = 12 (from Lab perceptual studies)
    if chroma < 12:
        return False

    hue = h * 2

    if 81 <= hue <= 140 and expected == "green":
        return True

    if (hue <= 15 or hue >= 330) and expected == "red":
        return True

    return False


# def _is_hue_in_range(hue, color_name):
#     """Check if hue falls within the expected range for the given color."""
#     if color_name == "green":
#         return 81 <= hue <= 140
#     elif color_name == "red":
#         # 0° to 10° and 355° to 360
#         return hue <= 15 or 330 <= hue
#     else:
#         return False

def is_gray(img, sat_threshold=60, gray_ratio=0.50):
    """
    Determines whether the image is mostly gray (low saturation).
    
    Args:
        img (np.ndarray): RGB or BGR NumPy image.
        sat_threshold (int): Max saturation for pixel to be considered gray (0–255).
        gray_ratio (float): Portion of low-sat pixels required to classify as gray.

    Returns:
        bool: True if mostly gray, False otherwise.
    """

    # Convert to HSV
    if img.shape[2] == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Image must have 3 channels.")

    S = hsv[:, :, 1]  # saturation

    # Count low-saturation (grayish) pixels
    gray_pixels = np.sum(S < sat_threshold)
    total_pixels = S.size

    ratio = gray_pixels / total_pixels
    # print(f"Gray pixels: {gray_pixels}, Total pixels: {total_pixels} Ratio: {ratio}")
    is_gray_ = ratio >= gray_ratio
    if is_gray_:
        return True
    else:
        return False
    
import cv2

def sec_trade_confirm(image, line_type, downsample, pt_1, pt_2):
    if image is None or image.size == 0:
        return True

    # Determine expected colors
    if "profit" in line_type or "tp" in line_type:
        expected_1 = "green"
    elif "loss" in line_type or "sl" in line_type:
        expected_1 = "red"
    else:
        expected_1 = "none"

    if "buy" in line_type and not ("sl" in line_type or "tp" in line_type):
        expected_2 = "green"
    elif "sell" in line_type and not ("sl" in line_type or "tp" in line_type):
        expected_2 = "red"
    elif "tp" in line_type:
        expected_2 = "green"
    elif "sl" in line_type:
        expected_2 = "red"
    else:
        expected_2 = "none"

    if expected_1 != expected_2:
        return True

    x1, y1 = pt_1
    x2, y2 = pt_2
    img_h, img_w = image.shape[:2]

    template_h = y2 - y1
    template_w = x2 - x1
    if template_h <= 0 or template_w <= 0:
        return True

    base_section1 = 23
    base_section2 = 15 if any(x in line_type for x in ["sell", "sl", "tp"]) else 14
    base_section3 = 14
    base_total = base_section1 + base_section2 + base_section3

    section1_width = int((23 / base_total) * template_w)
    section2_width = int((15 / base_total) * template_w)
    section3_width = int((14 / base_total) * template_w)

    def get_safe_crop_with_rect(img, top, bottom, left, right):
        t, b = max(0, top), min(img_h, bottom)
        l, r = max(0, left), min(img_w, right)
        if b <= t or r <= l:
            return None, None
        return img[t:b, l:r], (l, t, r, b)

    parts = {
        "Top": get_safe_crop_with_rect(
            image,
            y1 - template_h, y1,
            x1 + section3_width, x2
        ),
        "Bottom": get_safe_crop_with_rect(
            image,
            y2, y2 + template_h,
            x1 + section3_width, x2
        ),
        "Left": get_safe_crop_with_rect(
            image,
            y1, y2,
            x1 - (section2_width + section1_width), x1
        ),
        "Right": get_safe_crop_with_rect(
            image,
            y1, y2,
            x2+section1_width, x2 + (section2_width + section1_width + section1_width + 5)
        )
    }
    hue_matches = 0
    for name, (part, rect) in parts.items():
        if part is None or part.size == 0:
            continue
        
        # print("saving part: ",name)
        cv2.imwrite(f"{name}.png",part)

        try:
            if expected_1 == "green":
                green_per = green_percentage(part)
                if green_per > 70:
                    hue_match = True
                else:
                    hue_match = False
            
            elif expected_1 == "red":
                red_per = red_percentage(part)
                if red_per > 70:
                    hue_match = True
                else:
                    hue_match = False
            # part_fill= find_most_common_color(part)
            # # part_fill, color_per = find_most_common_color_and_percentage(part)
            # print("Side: ",name)
            # print("The part fill is: ",part_fill )
            # # print("The parts color percentage is:", color_per)
            # print("Expected color is: ",expected_1)
            # if part_fill is None:
            #     continue

            # hue1, s1, v1 = _hsv_from_rgb(part_fill)
            # hue_match = is_hue_in_range(hue1, s1, v1, expected_1)


            if hue_match:
            #     # 🔲 Draw rectangle on original image
            #     xL, yT, xR, yB = rect
            #     cv2.rectangle(
            #         image,
            #         (xL, yT),
            #         (xR, yB),
            #         (0, 255, 255),  # yellow for visibility
            #         2
            #     )
            #     cv2.putText(
            #         image,
            #         name,
            #         (xL, yT - 5),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         (0, 255, 255),
            #         1
            #     )
            #     from dump import display_image
            #     # print("The parts color percentage is:", color_per)
                

            #     display_image(image,"secondary trade confirmation")
                print(f"Match found! Part: {name} matches expected hue: {expected_1}")
                hue_matches = hue_matches+1
                if hue_matches > 0:
                    return False

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error analyzing {name} part: {e}")
            continue

    return True

def verify_trade_object_colors(full_image,pt_1,pt_2,cropped_img_bgr, line_type, resize_width=None, resize_height=None, downsample=2):
    """
    Verify fill and text color for sections 1 & 2 using hue-based classification.
    Works even if the template was resized proportionally.
    """
    # cv2.imshow("cropped_img_bgr",cropped_img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
    # img_rgb = cropped_img_bgr
    base_match = {"match":False} 
    


    # Base template dimensions
    base_section1 = 23
    base_section2 = 15 if ("sell" in line_type or "sl" in line_type or "tp" in line_type) else 14
    base_section3 = 14
    base_total = base_section1 + base_section2 + base_section3
    base_h = 14
        # Determine expected hue color type (red or green)
    if "profit" in line_type or "tp" in line_type:
        expected_1 = "green"
    elif "loss" in line_type or "sl" in line_type:
        expected_1 = "red"
    else:
        expected_1 = "none"

    # Section 2 expected color
    if "buy" in line_type and not ("sl" in line_type or "tp" in line_type):
        expected_2 = "green"
    elif "sell" in line_type and not ("sl" in line_type or "tp" in line_type):
        expected_2 = "red"
    elif "tp" in line_type:
        expected_2 = "green"
    elif "sl" in line_type:
        expected_2 = "red"
    else:
        expected_2 = "none"


    template_h , template_w = img_rgb.shape[:2]
    # print("The template hight ")
    # 6 = 13
    # x = template_height
    x = (template_h * 6 ) // 26
    # print("X: ",x)
    
    
    img_rgb=img_rgb[x:template_h-x, :]
    
    section1_width = int((23 / base_total) * template_w)
    section2_width = int((15 / base_total) * template_w)
    section3_width = int((14 / base_total) * template_w)


    section_1_fill = section1_width // 4
    section_2_fill = section1_width // 4
    section1 = img_rgb[:, :section_1_fill].copy()
    section2 = img_rgb[:, section1_width+section_2_fill :section1_width + section2_width - section_2_fill].copy()
    section3 = img_rgb[:,section1_width + section2_width:].copy()

    section1_fill = _most_common_color_from_array(section1, downsample_factor=downsample)
    hue1,s1,v1 = _hsv_from_rgb(section1_fill)
    s1_hue_match = is_hue_in_range(hue1,s1,v1, expected_1)
    if not s1_hue_match:
        return base_match
    section2_fill = _most_common_color_from_array(section2, downsample_factor=downsample)
    hue2,s2,v2 = _hsv_from_rgb(section2_fill)
    s2_hue_match = is_hue_in_range(hue2,s2,v2, expected_2)
    if not s1_hue_match:
        return base_match
    # section3_fill = _most_common_color_from_array(section3, downsample_factor=downsample)
    # hue3,s3,v3 = _hsv_from_rgb(section3_fill)



    s3_hue_match = is_gray(section3)
    if not s3_hue_match:
        return base_match



    s1_text = _black_white_percentage(section1)
    expected_1_text = "white" if "loss" in line_type else "black" 

    s1_text_match = (s1_text == expected_1_text)
    if not s1_text_match:
        return base_match

    overall_match = s1_hue_match and s2_hue_match and s1_text_match and s3_hue_match
    
    if overall_match:
        sec_check = sec_trade_confirm(full_image,line_type,downsample,pt_1,pt_2)
    else:
        sec_check = False


    return {
        "match": overall_match and sec_check,
        "primary_check": overall_match,
        "secondary_check": sec_check,
        "trade type": line_type,
        "section_1": {
            "expected_fill": expected_1,
            "actual_fill_rgb": section1_fill,
            "actual_fill_hue": hue1,
            "fill_match": s1_hue_match,
            "expected_text": expected_1_text,
            "actual_text": s1_text,
            "text_match": s1_text_match,
            "width": section1_width
        },
        "section_2": {
            "expected_fill": expected_2,
            "actual_fill_rgb": section2_fill,
            "actual_fill_hue": hue2,
            "fill_match": s2_hue_match,
            "width": section2_width
        },
        "section_3": {
            "expected_fill": "gray",
            "actual_fill_rgb": (0,0,0),
            "actual_fill_hue": (0,0,0),
            "fill_match": s3_hue_match,
            "width": section3_width
        }
    }

# if __name__ == "__main__":
#     color = verify_trade_object_colors(cv2.imread("sell_in_profit.png"),"sell_in_profit")
#     print(color)

def find_most_common_color(image):
    """
    Finds the most common color in a NumPy BGR image.
    """
    if image is None or image.size == 0:
        return None

    # Convert BGR (OpenCV) → RGB (normal)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to list of tuples
    pixels = rgb.reshape(-1, 3)
    pixels = [tuple(p) for p in pixels]

    color_counts = Counter(pixels)
    most_common = color_counts.most_common(1)

    return most_common[0][0] if most_common else None





def is_main_color_white(image: np.ndarray) -> float:
    """
    Calculates the percentage of white pixels in an image compared to the rest,
    using Otsu thresholding to separate bright and dark regions.

    Args:
        image (np.ndarray): Input image as a NumPy array.

    Returns:
        float: Percentage of white pixels (0 to 100).
    """
    try:
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Otsu threshold splits image into dark/light clusters
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Count white and dark pixels
        white_count = np.sum(binary > 240)
        total_count = binary.size

        # Calculate percentage of white pixels
        white_percentage = (white_count / total_count) * 100

        return white_percentage

    except Exception as e:
        print(f"Error processing image: {e}")
        return 0.0



def find_most_common_color_and_percentage(image_array: np.ndarray):
    """
    Finds the most common color in a NumPy image array and calculates its 
    percentage of the total pixels.

    Args:
        image_array (np.ndarray): A NumPy array representing an image. 
                                  Expected shape: (height, width, channels) or (pixels, channels).

    Returns:
        tuple: A tuple containing:
               - The most common color (tuple of integers, e.g., (BGR)).
               - The percentage of that color in the image (float).
    NOTE: This returns BGR
    """
    if image_array.ndim < 2:
        raise ValueError("Input array must have at least 2 dimensions (e.g., height, width or pixels, channels).")

    # 1. Reshape the image array to a list of pixels (flatten height and width)
    # The new shape will be (Total_Pixels, Channels)
    # Example: (480, 640, 3) becomes (307200, 3)
    pixels = image_array.reshape(-1, image_array.shape[-1])
    
    # Convert each color array (e.g., [10, 20, 30]) into an immutable tuple (10, 20, 30)
    # This is necessary because lists/arrays are not hashable and cannot be used as dictionary keys
    # for the Counter object.
    color_tuples = [tuple(color) for color in pixels]

    # 2. Count the frequency of each unique color
    color_counts = Counter(color_tuples)

    if not color_counts:
        # Handle case for an empty image array
        return None, 0.0

    # 3. Find the most common color and its count
    # `most_common(1)` returns a list of the 1 most common item: [((R, G, B), count)]
    most_common_color, count_of_most_common = color_counts.most_common(1)[0]

    # 4. Calculate the total number of pixels
    total_pixels = len(color_tuples)

    # 5. Calculate the percentage
    percentage = (count_of_most_common / total_pixels) * 100

    return most_common_color, percentage

def is_mostly_gray(img, sat_threshold=40, gray_ratio=0.75):
    """
    Determines whether the image is mostly gray (low saturation).
    
    Args:
        img (np.ndarray): RGB or BGR NumPy image.
        sat_threshold (int): Max saturation for pixel to be considered gray (0–255).
        gray_ratio (float): Portion of low-sat pixels required to classify as gray.

    Returns:
        bool: True if mostly gray, False otherwise.
    """

    # Convert to HSV
    if img.shape[2] == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Image must have 3 channels.")

    S = hsv[:, :, 1]  # saturation

    # Count low-saturation (grayish) pixels
    gray_pixels = np.sum(S < sat_threshold)
    total_pixels = S.size

    ratio = gray_pixels / total_pixels
    print(f"Gray pixels: {gray_pixels}, Total pixels: {total_pixels} Ratio: {ratio}")
    return ratio >= gray_ratio

if __name__ == "__main__":
    # image = cv2.imread("color.png")
    # import time 
    # start = time.time()
    # color,per = find_most_common_color_and_percentage(image)
    # print(f"This took: {time.time() - start}")
    # start_2 = time.time()
    # print("Color: ",color ,"Percentage: ",per)
    # main_color = _most_common_color_from_array(image)
    # print(f"The second function took: {time.time() - start_2}")
    # print("Most common color: ",main_color)
    # (np.uint8(178), np.uint8(165), np.uint8(206))

    # h,s,v = _hsv_from_rgb([206,165,178])
    # print("HSV: ",h,s,v)
    # print(is_hue_in_range(h,s,v,"red"))
    for i in ("Top","Bottom","Left","Right"):
        image = cv2.imread(f"{i}.png")
        # c1 ,p1 = find_most_common_color_and_percentage(image)
        # c2 = find_most_common_color(image)
        import time
        start_1 = time.time()
        per = green_percentage(image)
        print("The first percetage function took: ",time.time() - start_1)
        # start_2 = time.time()
        # per_2 = red_percentage_2(image)
        # print("The second percentage function took: ",time.time() - start_2)
        print("Side; ",i)
        print("Percentage: ",per)
        # print("Percentage 2: ",per_2)
        # print(f"c1: {c1}")
        # print(f"c2: {c2}")
        print("\n")
    # print(is_hue_in_range(0, 255, 255, "red"))   # should be True
    # print(is_hue_in_range(60, 255, 255, "red"))  # False
    # print(is_hue_in_range(0, 10, 255, "red"))    # False (grayish)
