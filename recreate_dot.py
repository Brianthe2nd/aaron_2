import cv2
import numpy as np
import csv
import os
import glob
import pandas as pd
from pair import get_pairs
from config import get_config,update_config
from search import get_pair_search_img
from logo import detect_best_logo_height


def calculate_green_dot(img, rect):
    """
    Crop a region around the detected green dot that is at least
    as large as the template image ("green_dot_centered.png").
    """
    x_or, y_or, w_or, h_or = rect

    # Load the template and get its size
    temp_img = cv2.imread("green_dot_centered.png")
    temp_h, temp_w = temp_img.shape[:2]

    # --- Compute the desired crop size ---
    crop_w = max(w_or, temp_w)
    crop_h = max(h_or, temp_h)

    # Center the crop on the detected rectangle
    cx = x_or + w_or // 2
    cy = y_or + h_or // 2

    # Compute the top-left corner of the crop so the crop is centered
    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Clamp coordinates to stay inside image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    # Crop the region
    # green_img = img[y1:y2, x1:x2]

    return (y1,y2,x1,x2)

from PIL import ImageFont, ImageDraw, Image

def get_text_bbox(text: str, font_path: str = "trebuchet-ms-2/trebuc.ttf", font_size: int = 14):
    """
    Returns the width and height of a text bounding box using a given font.
    
    :param text: Text to measure
    :param font_path: Path to .ttf or .otf font
    :param font_size: Font size in px
    :return: (width, height)
    """
    # Load font
    font = ImageFont.truetype(font_path, font_size)

    # Dummy image needed for drawing context
    temp_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(temp_img)

    # Use textbbox for accurate measurement
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)

    width = right - left
    height = bottom - top

    return width, height


def get_x_minus(logo_size):
    """133.jpg"""
    """$ py border.py
    LOWEST
    Width: 45 Height: 10
    The border is :  75

    $ py border.py
    HIGHEST
    Width: 76 Height: 10
    The border is :  106"""
    interpunct = 10
    behind = 12
    front = 8
    space = interpunct + front + behind
    """space = 30 logo_size = 39 font_size = 13"""
    
    print("Logo size: ",logo_size)
    space = (logo_size * 106) / 39
    return int(space)

# def find_x_divider(point,text_start,x_minus):
#     # ((274, 156), (286, 167))
#     x1,y1 = point[0]
#     x2,y2 = point[1]
    
#     """133.jpg"""
#     interpunct = 10
#     behind = 12
#     front = 8
#     space = interpunct + front + behind
#     """space = 30 logo_size = 39 font_size = 13"""
#     print("text_start: ",text_start)
#     print("x1: ",x1)
#     print("xminus: ",x_minus)
#     x_divider = ((x1 - x_minus) + text_start) - front 

#     return x_divider
def find_x_divider(point,text_start,x_minus):
    # ((274, 156), (286, 167))
    x1,y1 = point[0]
    x2,y2 = point[1]
    
    """133.jpg"""
    interpunct = 10
    behind = 12
    front = 8
    space = interpunct + front + behind
    """space = 30 logo_size = 39 font_size = 13"""
    print("text_start: ",text_start)
    print("x1: ",x1)
    print("xminus: ",x_minus)
    x_divider = ((x1 - x_minus) + text_start) - front 

    return x_divider


def get_pair_img(point,image,x_minus):
    x_1 = point[0][0]
    x_2 = point[1][0]
    
    y_1 = point[0][1]
    y_2 = point[1][1]
    
    sym_x_1 = x_1 - x_minus
    sym_x_2 = x_1
    
    sym_y_1 = y_1 - 10
    sym_y_2 = y_2 + 10
    
    return image[sym_y_1:sym_y_2 , sym_x_1:sym_x_2]

def post_process_points(arr, k=5, max_cleaned=4):
    if len(arr) <= 3:
        return arr[:max_cleaned]
    arr.sort(key = lambda x: x[0])
    # Step 1: compute diffs
    diffs = [arr[i+1][0] - arr[i][0] for i in range(len(arr)-1)]

    # Step 2: robust typical difference
    # import numpy as np
    median_diff = np.median(diffs)
    # print("DIFFS")
    # print(diffs)
    # print("median_diff: ",median_diff)
    # # Step 3: find boundaries
    # print("Diff: ",k * median_diff)
    boundaries = [0]  # first segment always starts at index 0
    for i, d in enumerate(diffs):
        if d > k * median_diff:
            # print(f"D: {d}")
            boundaries.append(i+1)

    # Step 4: collect cleaned points (first of each segment)
    cleaned = []
    for b in boundaries:
        cleaned.append(arr[b])
        if len(cleaned) >= max_cleaned:
            break

    return cleaned

def find_green_dot(image,logo_loc,image_path = "image"):
    
    h,w = image.shape[:2]
    top_left , bottom_right = logo_loc
    # print("logo location is")
    # print(logo_loc)
    # print("This rectangle is from green_dot for logo location")
    # cv2.rectangle(image,top_left,bottom_right,(255,0,0),2)
    logo_bottom_y = bottom_right[1]
    logo_top_y = top_left[1]
    logo_bottom_x = bottom_right[0]
    logo_size = abs(logo_top_y - logo_bottom_y)
    # print("logo_size: ",logo_size)
    # 66 = 91
    # logo_size
    actual_top = (logo_size * 91 )//66
    actual_bottom = (logo_size * 113 )//66
    add = (logo_size * 48) // 20
    search_bottom = (logo_size * 53) // 66
    search_to_top = actual_top - search_bottom 
    y = max(logo_bottom_y + actual_top - add , search_bottom + logo_bottom_y , (search_to_top // 2) + logo_bottom_y + search_bottom )
    h = abs(abs(actual_top - y) + search_bottom) // 2
    rect = (0 ,y, w , h) 

    x, y, w, h = rect
    # cv2.rectangle(image,(0,y),(w,y+h),(0,255,0),2)
    # cv2.imshow("tf",image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


    return rect

def is_in_first_section(x,w):
    first_section = w//4
    if x <= first_section:
        return True
    else:
        return False

def points_on_the_same_level(points):
    y_1 = points[0][0][1]
    y_2 = points[1][0][1]
    
    if abs(y_1- y_2) <=  5:
        return True
    else:
        return False



def process_multi_points(pair_objs, image, screens_data,logo_size,logo_loc,trader,y=10):
    # cv2.imshow("multi point image",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    """
    Process a list of points to extract screen pair data.

    Args:
        points (list): List of points (each a tuple or list of coordinates).
        image (numpy.ndarray): The main image being analyzed.
        screens_data (list): A list to which processed screen data will be appended.

    Returns:
        list: Updated screens_data with new screen info.
    """
    print("We have more than one screen")
    if not pair_objs:
        # print("No points provided.")
        return screens_data
    logo_template = cv2.imread("templates/x_logo.png")
    # logo_size,_ = detect_best_logo_height(image,logo_template)
    x_minus = get_x_minus(logo_size)
    pair_search_img = get_pair_search_img(image,logo_size,logo_loc,name = trader)

    # _,pair_search,_,_,is_micro = get_pair(pair_search_img)
    print("The trader in multi points is ")
    search_res = get_pairs(pair_search_img,search = True,trader = trader,logo_height=logo_size, orig_y = y)
    pair_search = search_res.get("mapped")
    if pair_search and pair_search.startswith("I"):
        pair_search = pair_search[1:]
    # print(f"The pair search is: {pair_search}")
    
    print(f"Search pair: {pair_search}")
    print("PAIR OBJECTS")
    print(pair_objs)
    pairs = []
    # x_dividers = []
    unmapped=0
    for i, pair_dict in enumerate(pair_objs):
        point = None
    #     # {"extracted":expected,"mapped": mapped"x_start":fx"fuzzy":True"is_micro":is_micro}
        
        unmapped_screen_pair = pair_dict["extracted"] 
        pair = pair_dict["mapped"]
        text_start = pair_dict["x_start"]
        text_match = pair_dict["fuzzy"]
        is_micro = pair_dict["is_micro"]

    #     if not text_match:
    #         if len(pair) == 2: 
    #             pair = pair_search
    #         else:
    #             unmapped = unmapped + 1
    #             unmapped_data = {"point":point,"text_start":text_start,"x_minus":x_minus,"unmapped_screen_pair":unmapped_screen_pair}
             
    #     # x_divider = find_x_divider(point,text_start,x_minus)
        x_divider = text_start - 10
        # print(f"Screen pair: {pair}")
        pairs.append((pair, point ,x_divider))
    
     # Match detected pairs with the search pair
    print("PAIRS: ")
    print(pairs)
    matched_pairs = [p for p in pairs if p[0] == pair_search]
    
    # if unmapped == 1 and not matched_pairs:
    #     for pair in pairs :
    #         if pair[0] == unmapped_data.get("unmapped_screen_pair"):
    #             pair[0] = pair_search
                
            
    if matched_pairs:
        # Create s;tructured data for all points
        for i, (pair, point, x_divider) in enumerate(pairs):
            # x_divider = find_x_divider(image,point[0][0])
            screen_data = {"pair": pair, "x_divider": x_divider ,"pred": False}
            screens_data.append(screen_data)
            # pprint(screen_data)
    else:
        print("There is an unknown screen pair")

    return screens_data



def process_points(image,top_rect,logo_size,logo_loc,trader):
    # from dump import display_image
    # display_image(image,"process points")
    # cv2.imshow("multi point image in process points",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image_height,image_width = image.shape[:2]
    x, y, w, h = top_rect
    cropped_image = image[y:y+h,x:w]
    # display_image(cropped_image,"process points")
    # print("THE TRADER WE ARE PROCESSING IS: ",trader)
    pairs = get_pairs(cropped_image,search = False,trader = trader,logo_height=logo_size , orig_y = y)
    # {"extracted":expected,"mapped": mapped"x_start":fx"fuzzy":True"is_micro":is_micro}
    print("pairs")
    print(pairs)
    pairs.sort(key=lambda item: item["x_start"])

    screens_data =[]
    if len(pairs) != 0:
        if len(pairs) == 1:
            pair = pairs[0]
            print("We have one screen")
            x_minus = get_x_minus(logo_size)
            # screen_pair_img = get_pair_img(points[0],image,x_minus)
            # unmapped_screen_pair,screen_pair,text_start,_,is_micro = get_pair(screen_pair_img)
            # x_divider = find_x_divider(point,text_start,x_minus)
            text_start = pair["x_start"]
            x_divider = text_start - 10
            pair_search_img = get_pair_search_img(image,logo_size,logo_loc,trader )
            # _,search_pair,_,_,is_micro = get_pair(pair_search_img)
            # if search_pair and search_pair.startswith("I"):
            #     search_pair = search_pair[1:]
            search_res = get_pairs(pair_search_img,search = True,trader = trader,logo_height=logo_size, orig_y = y)
            search_pair = search_res.get("mapped")
            if search_pair is None:
                return screens_data
            # search_pair.sort(key=lambda item: item["x_start"]
            screen_pair = pair["mapped"]
            print(f"Screen pair: {pair["mapped"]}")
            print(f"Search pair: {search_pair}")
            if screen_pair == search_pair:
                pair_name = screen_pair
                screen_data = {"pair": pair_name,"x_divider": x_divider , "pred": False}
                screens_data.append(screen_data)
            else:
                # print(f"The x section point is : {points[0][0][0]}")
                if is_in_first_section(text_start,image_width):
                    print("It is in the first section")
                    screen_data = {"pair":screen_pair,"x_divider":x_divider,"pred": False}
                    screen_2_data = {"pair":search_pair,"x_divider":int(image_width/2),"pred": True}
                    screens_data.append(screen_data)
                    screens_data.append(screen_2_data)
                else:
                    print("The point is in the second section")
                    # x_divider = int(image_width/2)
                    screen_2_data = {"pair":search_pair,"x_divider":0,"pred": True}                    
                    screen_data = {"pair":screen_pair,"x_divider":x_divider,"pred": False}
                    screens_data.append(screen_data)
                    screens_data.append(screen_2_data)

            
            # pprint(screen_data)
            
            """confirm if the pair is the same as the search bar option"""
            """if yes then there is one page """
            """if no then there are two pages and dee has actally clicked on one of them , default the x_division to half of the page"""
        
        else:
            screens_data = process_multi_points(pairs,image,screens_data,logo_size,logo_loc,trader,y)
    else:
        pair_search_img = get_pair_search_img(image,logo_size,logo_loc,trader)

        search_res = get_pairs(pair_search_img,search = True,trader = trader,logo_height=logo_size, orig_y = y)
        print("search res: ",search_res)
        search_pair = search_res.get("mapped")
        if search_pair is None:
            return screens_data
        screen_data = {"pair": search_pair, "x_divider": 0 ,"pred": False}
        screens_data.append(screen_data)
        print("The number of screen pairs is zero")
        print("predicted screen data is: ")
        print(screens_data)

    return screens_data

