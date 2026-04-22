
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import cv2
from similar import similarity
from pprint import pprint
from resize import resize_proportional
from color import verify_trade_object_colors
from face import find_camera_box
import time
import numpy as np


def create_trade_object(ticks_size = 8 ,padding = 3 ,_1_size =9 ,line_type = "buy_in_profit"):
    font_path = "arial.ttf"

    height = 14
    section1_width = 23
    section2_width = 14
    section3_width = 14
    border_width = 1

    red_fill = (255,0,0)
    grey_fill = (211,215,218)
    green_fill = (0,255,0)
    red_border_color = (161,15,19) # 50% opacity red
    background_color = (30, 30, 30, 255) # dark background
    grey_border_color = (105,107,110)
    green_border_color = (23, 159, 28)
    white = (255,255,255)
    black = (0,0,0)
    text_color = black
    section_2_text = "3"
    if "buy" in line_type and "tp" not in line_type and "sl" not in line_type:
        section_2_fill = green_fill
        section_2_border_color = green_border_color
    
    
    if "sell" in line_type and "tp" not in line_type and "sl" not in line_type:
        section_2_fill = red_fill
        section_2_border_color = red_border_color
        section2_width = 15
        section_2_text = "-3"
    
    
    if "profit" in line_type:
        section_1_fill = green_fill
        section_1_border_color = green_border_color
        
    if "loss" in line_type:
        section_1_fill = red_fill
        section_1_border_color = red_border_color
        text_color = white
    
    if "buy" in line_type and ("sl" in line_type or "tp" in line_type):
        section2_width = 15
        section_2_text = "-3"
    
    if "sl" in line_type:
        section_1_fill = red_fill
        section_1_border_color = red_border_color
        section_2_fill = red_fill
        section_2_border_color = red_border_color
    
    if "tp" in line_type:
        section_1_fill = green_fill
        section_2_fill = green_fill
        section_1_border_color = green_border_color
        section_2_border_color = green_border_color
    
    total_width = section1_width + section2_width + section3_width
    object_img = Image.new("RGBA", (total_width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(object_img)

    draw.rectangle([0, 0, section1_width - 1, height - 1], fill=section_1_fill)
    draw.rectangle([section1_width, 0, section1_width + section2_width - 1, height - 1], fill=section_2_fill)
    x3_start = section1_width + section2_width
    draw.rectangle([x3_start, 0, x3_start + section3_width - 1, height - 1], fill=grey_fill)

    if height >= 2:
        # section_1
        draw.line([(0, 0), (section1_width - 1, 0)], fill=section_1_border_color, width=1)
        draw.line([(0, height - 1), (section1_width - 1, height - 1)], fill=section_1_border_color, width=1)
        draw.line([(section1_width - 1, 0), (section1_width - 1, height - 1)], fill=section_1_border_color, width=1)
        
        #section 2
        draw.line([(section1_width, 0), (x3_start - 1, 0)], fill=section_2_border_color, width=1)
        draw.line([(section1_width, height-1), (x3_start - 1, height-1)], fill=section_2_border_color, width=1)
        draw.line([(x3_start - 1, 0), (x3_start - 1, height - 1)], fill=section_2_border_color, width=1)
        draw.line([(section1_width, 0), (section1_width, height - 1)], fill=section_2_border_color, width=1)
        
        #section 3
        draw.line([(total_width - 1, 0), (total_width - 1, height - 1)], fill=grey_border_color, width=1)
        draw.line([(x3_start, 0), (x3_start, height - 1)], fill=grey_border_color, width=1)
        draw.line([(x3_start, 0), (total_width, 0)], fill=grey_border_color, width=1)
        draw.line([(x3_start, height-1), (total_width, height-1)], fill=grey_border_color, width=1)

    def draw_centered_text(draw, text, font, section_start, section_width, section_height, fill, x_offset=None):
        """Helper to center text inside a rectangular section.
        If x_offset is provided, it overrides horizontal centering."""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_y = (section_height - text_h) // 2 - bbox[1]
        if x_offset is not None:
            text_x = section_start + x_offset
        else:
            text_x = section_start + (section_width - text_w) // 2 - bbox[0]
        draw.text((text_x, text_y), text, fill=fill, font=font)
    try:
        _1_font = ImageFont.truetype(font_path, _1_size)
        ticks_font = ImageFont.truetype(font_path, ticks_size)
    except IOError:
        print("⚠️ 'arial.ttf' not found. Falling back to default font.")
        _1_font = ImageFont.load_default()
        ticks_font = ImageFont.load_default()

    line_color = (40,44,45) 
    line_width = 1
    x3_left = x3_start + padding
    x3_right = x3_start + section3_width - 1 - padding
    y_top = 0 + padding
    y_bottom = height - 1 - padding
    draw.line([(x3_left, y_top), (x3_right, y_bottom)], fill=line_color, width=line_width)
    draw.line([(x3_left, y_bottom), (x3_right, y_top)], fill=line_color, width=line_width)
    draw_centered_text(draw, section_2_text, _1_font, section1_width, section2_width, height, black)
    draw_centered_text(draw, "ticks", ticks_font, 0, section1_width, height, text_color, x_offset=2)
    out_path = "trade_object_on_dark_bg.png"
    object_img.save(out_path)
    return cv2.imread(out_path)

def undo_rotate_180(point, width, height):
    
    x, y = point

    # Undo 180 rotation: same formula as rotate, because 180° twice = identity
    normal_x = width - 1 - x
    normal_y = height - 1 - y

    return normal_x, normal_y


def point_in_face_box(image, face_boxes, pt, point_height, point_width):
    """
    expects pt to be in the format (x,y)
    Returns True if the object (defined by top-left pt + size)
    intersects any face bounding box.
    """

    if not face_boxes:
        return False

    x1, y1 = pt
    img_h, img_w = image.shape[:2]

    # Undo rotation (your original function call)
    x1, y1 = undo_rotate_180((x1, y1), img_w, img_h)

    # Object bounding box
    obj_x1 = x1
    obj_y1 = y1
    obj_x2 = x1 + point_width
    obj_y2 = y1 + point_height

    for (fx1, fy1, fx2, fy2) in face_boxes:

        # ---------------------------
        # RECTANGLE INTERSECTION TEST
        # ---------------------------
        overlap_x = not (obj_x2 < fx1 or obj_x1 > fx2)
        overlap_y = not (obj_y2 < fy1 or obj_y1 > fy2)

        if overlap_x and overlap_y:
            return True  # object touches or enters face area

    return False

def get_resize_height(logo_height):
    # 39 = 17
    # 49 = 20 
    # logo_height =
    return int((logo_height * 20) / 49)




def fetch_trades(image, logo_height, logo_loc): 
    # from dump import display_image
    # display_image(image,"fetch trades image")
    total_start = time.time()
    
    # --- Initialization ---
    top_left, bottom_right = logo_loc
    bottom_right_y = bottom_right[1]
    hi, wi = image.shape[:2]
    
    face_box = find_camera_box(image)
    image = cv2.rotate(image, cv2.ROTATE_180)
    
    confirmed_tt = []
    found_main_trade = False 
    trade_types = ["buy_in_loss", "sell_in_profit", "sell_in_loss", "buy_in_profit", "sl", "tp"]
    resize_height = get_resize_height(logo_height)
    threshold = 0.5
    min_dist = 2  # Suppression radius

    # --- Stage 1: Main trade detection ---
    for trade_type in trade_types:
        if found_main_trade or trade_type in ["tp", "sl"]:
            continue

        type_start = time.time()
        trade_object = create_trade_object(line_type=trade_type)
        trade_object = resize_proportional(trade_object, height=resize_height)
        th, tw = trade_object.shape[:2]
        trade_object = cv2.rotate(trade_object, cv2.ROTATE_180)
        
        result = cv2.matchTemplate(image, trade_object, cv2.TM_CCOEFF_NORMED)
        
        # --- OPTIMIZED SUPPRESSION ---
        loc = np.where(result >= threshold)
        scores = result[loc]
        # Sort by match quality (best first)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Use a boolean mask to "visit" pixels - much faster than linalg.norm
        mask = np.zeros(result.shape, dtype=bool)
        matched_points = []

        for idx in sorted_indices:
            pt_y, pt_x = loc[0][idx], loc[1][idx]
            
            if mask[pt_y, pt_x]:
                continue
            
            # Mark neighborhood as visited
            y1, y2 = max(0, pt_y - min_dist), min(result.shape[0], pt_y + min_dist + 1)
            x1, x2 = max(0, pt_x - min_dist), min(result.shape[1], pt_x + min_dist + 1)
            mask[y1:y2, x1:x2] = True
            
            # Logic constraints
            rotated_y = abs(hi - pt_y - th)
            if not point_in_face_box(image, face_box, (pt_x, pt_y), th, tw) and rotated_y > bottom_right_y:
                cropped = image[pt_y:pt_y+th, pt_x:pt_x+tw]
                
                cropped = cv2.rotate(cropped, cv2.ROTATE_180)
                
                trade_c = verify_trade_object_colors(image, (pt_x,pt_y),(pt_x+tw ,pt_y+th),cropped, trade_type, resize_height=resize_height)
                # print(trade_c)
                if trade_c["match"]:
                    # cv2.rectangle(image, (pt_x,pt_y),(pt_x+tw ,pt_y+th), (0,255,0) , 1)
                    matched_points.append((pt_x, pt_y))
                    if trade_type in ["buy_in_loss", "sell_in_loss", "sell_in_profit"]:
                        found_main_trade = True
                        break
        
        # Second-stage 10px filter (only on a few points, so standard loop is fine here)
        refined_points = []
        for pt in matched_points:
            if all(np.linalg.norm(np.array(pt) - np.array(fp)) > 10 for fp in refined_points):
                refined_points.append(pt)
        
        confirmed_tt.extend([trade_type] * len(refined_points))
        print(f"[TIMER] {trade_type} took: {time.time() - type_start:.4f}s")
    # from dump import display_image
    # display_image(image)
    # --- Stage 2: SL / TP Detection ---
    tp = False
    sl = False
    if found_main_trade or "buy_in_profit" in confirmed_tt:
        check_tp = (confirmed_tt.count("buy_in_profit") > 1) if "buy_in_profit" in confirmed_tt else True

        # --- SL CHECK ---
        sl_start = time.time()
        sl_obj = create_trade_object(line_type="sl")
        sl_obj = resize_proportional(sl_obj, height=resize_height)
        sl_obj = cv2.rotate(sl_obj, cv2.ROTATE_180)
        res_sl = cv2.matchTemplate(image, sl_obj, cv2.TM_CCOEFF_NORMED)
        
        sl_loc = np.where(res_sl >= threshold)
        sl_mask = np.zeros(res_sl.shape, dtype=bool)
        sl_th, sl_tw = sl_obj.shape[:2]
        
        for idx in np.argsort(res_sl[sl_loc])[::-1]:
            py, px = sl_loc[0][idx], sl_loc[1][idx]
            if sl_mask[py, px]: continue
            sl_mask[max(0, py-min_dist):py+min_dist+1, max(0, px-min_dist):px+min_dist+1] = True
            
            if not point_in_face_box(image, face_box, (px, py), sl_th, sl_tw) and abs(hi - py - sl_th) > bottom_right_y:
                cropped = image[py:py+sl_th, px:px+sl_tw]
                
                cropped = cv2.rotate(cropped, cv2.ROTATE_180)
                if verify_trade_object_colors(image,(px,py),(px+sl_tw ,py+sl_th),cropped, "sl", resize_height=resize_height)["match"]:
                    # cv2.rectangle(image, (px,py),(px+sl_tw ,py+sl_th), (0,255,0) , 1)
                    sl = True
                    break
        print(f"[TIMER] SL Stage took: {time.time() - sl_start:.4f}s")

        # --- TP CHECK ---
        if check_tp:
            tp_start = time.time()
            for trade_type in ["tp", "buy_in_profit"]:
                tp_obj = create_trade_object(line_type=trade_type)
                tp_obj = resize_proportional(tp_obj, height=resize_height)
                tp_obj = cv2.rotate(tp_obj, cv2.ROTATE_180)
                res_tp = cv2.matchTemplate(image, tp_obj, cv2.TM_CCOEFF_NORMED)
                
                tp_loc = np.where(res_tp >= threshold)
                tp_mask = np.zeros(res_tp.shape, dtype=bool)
                tph, tpw = tp_obj.shape[:2]
                
                for idx in np.argsort(res_tp[tp_loc])[::-1]:
                    py, px = tp_loc[0][idx], tp_loc[1][idx]
                    if tp_mask[py, px]: continue
                    tp_mask[max(0, py-min_dist):py+min_dist+1, max(0, px-min_dist):px+min_dist+1] = True
                    
                    if not point_in_face_box(image, face_box, (px, py), tph, tpw) and abs(hi - py - tph) > bottom_right_y:
                        cropped = image[py:py+tph, px:px+tpw]
                        
                        cropped = cv2.rotate(cropped, cv2.ROTATE_180)
                        if verify_trade_object_colors(image, (px,py),(px+tpw ,py+tph),cropped, trade_type, resize_height=resize_height)["match"]:
                            # cv2.rectangle(image, (px,py),(px+tpw ,py+tph), (0,255,0) , 1)
                            tp = True
                            break
                if tp: break
            print(f"[TIMER] TP Stage took: {time.time() - tp_start:.4f}s")

    
     
    # --- Final Mapping ---
    main_trade, status = "unknown", None
    if "buy_in_loss" in confirmed_tt: main_trade, status = "buy", "loss"
    elif "sell_in_loss" in confirmed_tt: main_trade, status = "sell", "loss"
    elif "sell_in_profit" in confirmed_tt: main_trade, status = "sell", "profit"
    elif "buy_in_profit" in confirmed_tt: main_trade, status = "buy", "profit"

    if "buy_in_profit" in confirmed_tt:
        tp = confirmed_tt.count("buy_in_profit") > 1

    print(f"*** TOTAL EXECUTION TIME: {time.time() - total_start:.4f}s ***")
    trade_d = {"trade_type": main_trade, "status": status, "sl": sl, "tp": tp}
    print(trade_d)
    # cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
    # cv2.imshow("frame",image)
    # # cv2.resizeWindow("frame",1280,720)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # from dump import display_image
    # display_image(image)
    return trade_d

if __name__ == "__main__":
    # trade_types = ["buy_in_profit", "buy_in_loss", "sell_in_profit", "sell_in_loss",
    #                "buy_sl", "sell_sl", "buy_tp", "sell_tp"]
    from logo import detect_best_logo_height
    image = cv2.imread("screen_0.png")
    # frame_image = cv2.imread("frame.jpg")
    # logo_height = 66
    # logo_template = cv2.imread("templates/x_logo.png")
    # logo_size,_,logo_loc = detect_best_logo_height(frame_image,logo_template)
    # {'logo_height': 41.6, 'logo_top_left': (178, 51), 'logo_bottom_right': (219, 92)}
    logo_size = 41.6
    logo_loc = ((178, 51),(219, 92))
    trades = fetch_trades(image,logo_size,logo_loc)
    print(trades)
    # match_image = cv2.imread("buy_in_profit.png")  
    # match_fail_image = cv2.imread("surviving_crop_138_277.png")
