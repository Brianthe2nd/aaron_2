
import os
import cv2
import numpy as np
import json
import threading
import time
import csv
import re
from typing import Optional
import os

from image_processing import crop_chart_region,pre_process
from color import process_color,is_main_color_white
from gemini import get_levels
from mt5_functions import recalculate_risk,update_trade,open_trade,close_trade
import traceback
from mt5_functions import map_pairs
from std_out import Print,play_error_sound,log_exception
from logo import check_logo,detect_best_logo_height
from name import get_trader_name
from screen import capture_screen
from color import blue_percentage
from std_out import Print
from all_data import get_data
# from image_processing import crop_right
from easy_boxes import in_sym_search
from cropcross import crop_right
from config import get_config,update_config
from face import detect_faces, detect_faces_text

box_3_x = 0
box_3_y = 1
box_2_x = 2
box_2_y = 3 
box_1_x = 4
box_1_y = 5
box_height = 6
box_width = 7
box_val = 8
CSV_FILE = os.path.join(os.path.dirname(__file__), "trades_2_log.csv")

def reduce_X_close_points_exact(points, threshold=3 ,y_threshold=20):
    reduced = []
    used = [False] * len(points)

    for i, (x1, y1) in enumerate(points):
        if used[i]:
            continue
        
        group = [(x1, y1)]
        # reduced.append((x1, y1))
        used[i] = True

        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            if not used[j] and abs(x1 - x2) < threshold and abs(y1 - y2) < y_threshold:
                used[j] = True
                group.append((x2, y2))
        leftmost = min(group, key=lambda p: p[0])
        reduced.append(leftmost)

    return reduced

def reduce_Y_close_points_exact(points, threshold=3, x_threshold=20):
    reduced = []
    used = [False] * len(points)

    for i, (x1, y1) in enumerate(points):
        if used[i]:
            continue

        group = [(x1, y1)]
        used[i] = True

        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            if not used[j] and abs(y1 - y2) < threshold and abs(x1 - x2) < x_threshold:
                group.append((x2, y2))
                used[j] = True

        # Keep only the leftmost point (smallest x)
        leftmost = min(group, key=lambda p: p[0])
        reduced.append(leftmost)

    return reduced


def match_at_scale(image, original_template, scale, mode, threshold, all_points_lock, all_points):
    template = cv2.resize(original_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = template.shape[:2]

    if h > image.shape[0] or w > image.shape[1]:
        return

    result = cv2.matchTemplate(image, template, mode)
    is_low_better = mode in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    if is_low_better:
        locations = np.where(result <= threshold)
    else:
        locations = np.where(result >= threshold)

    scale_points = [(pt[0], pt[1], w, h, scale) for pt in zip(*locations[::-1])]

    # Use lock to safely append to shared list
    with all_points_lock:
        all_points.extend(scale_points)


def match_template_and_draw(image, template_path, threshold=0.9, mode=cv2.TM_CCOEFF_NORMED ,check_x_scale = False):
    
    if not check_x_scale:
        if not os.path.exists(template_path):
            return image, []  # Return original image and no matches

        template = cv2.imread(template_path)
        if template is None:
            return image, []

        h, w = template.shape[:2]
    

        result = cv2.matchTemplate(image, template, mode)

        # Determine matching logic based on mode
        is_low_better = mode in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
        if is_low_better:
            locations = np.where(result <= threshold)
        else:
            locations = np.where(result >= threshold)

        matches = []
        points=[]
        for pt in zip(*locations[::-1]):  # (x, y)
            x = pt[0]
            y = pt[1]
            points.append((x, y))
        
        points = reduce_X_close_points_exact(points,y_threshold=w/2) 
        points = reduce_Y_close_points_exact(points, x_threshold=w*5)   
        
        for pt in zip(*locations[::-1]):  # (x, y)
            match_val = result[pt[1], pt[0]]
            x= pt[0]
            y= pt[1]
            if (x, y) in points:
                matches.append((x , y , x-w , y , x-3*w , y , match_val))
                # Draw rectangle on the image
                # top_left = (x, y)
                # bottom_right = (x + w, y + h)
                # cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 1)
            else: 
                continue



        return image, matches,w,h
    else:
        if not os.path.exists(template_path):
            return image, []

        original_template = cv2.imread(template_path)
        if original_template is None:
            return image, []

        matches = []
        all_points = []
        all_points_lock = threading.Lock()
        # check_config
        threads = []
        for scale in np.linspace(0.6, 1.7, 40):
            t = threading.Thread(target=match_at_scale, args=(image, original_template, scale, mode, threshold, all_points_lock, all_points))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        w=0
        h=0
        seen = set()
        unique = []

        for item in all_points:
            key = (item[0], item[1])  # only compare first two values
            if key not in seen:
                seen.add(key)
                unique.append(item)
        all_points =[]
        all_points.extend(unique)
        if all_points:
            raw_points = [(x, y) for x, y, w, h, _ in all_points]
            w = all_points[0][2]
            h = all_points[0][3]
            raw_points = reduce_X_close_points_exact(raw_points, y_threshold=w / 2)
            raw_points = reduce_Y_close_points_exact(raw_points, x_threshold=w * 5)
            for x, y, w, h, val in all_points:
                if (x, y) in raw_points:
                    matches.append((x, y, x - w - 1, y, x - 3 * w, y, h , w, val))


        Print("The number of matches is :", len(matches))
        """update config"""
        return image, matches, w, h if all_points else (image, [], 0, 0)



# --- Main testing code ---

def get_dominant_color_name(color_percentages):
    # Find the color with the highest percentage
    dominant_color = max(color_percentages, key=color_percentages.get)
    if color_percentages[dominant_color] > 50:
        return dominant_color  # e.g., 'red', 'green', or 'gray'
    return None  # No color dominant enough
    



def save_trade_event(pair, trader, event_type, timestamp, time_s, video_link):
    frame_number_file = os.path.join(os.path.dirname(__file__),"frame_number.txt")
    screen_num_file = os.path.join(os.path.dirname(__file__), "screen_num.txt")
    screen_num = read_file(screen_num_file)
    frame_num = read_file(frame_number_file)
    info_file = os.path.join(os.path.dirname(__file__), "info.json")
    with open(info_file,"r") as file:
        info = json.load(file)
    
    
    video_link = info.get("video_link")
    video_title = info.get("video_title") 
    # frame_num = rea
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([pair, trader, event_type, timestamp, time_s, video_link , screen_num , frame_num , video_title])
        
def get_latency_time(mode="normal"):
    latencies = {
        "normal": (15,30),
        "low": (5,15),
        "ultra": (2,5)
    }
    return latencies.get(mode.lower(), "[Unknown mode]")


def get_level_data(frame, trades, start, stream_mode="normal"):
    try:
        trade_json = get_levels(frame, trades)

        # Remove Markdown JSON fences if present
        if trade_json.startswith("```"):
            trade_json = re.sub(r"^```[a-zA-Z0-9]*\s*", "", trade_json)  # remove opening ```
            trade_json = re.sub(r"\s*```$", "", trade_json)              # remove closing ```

        trade_json = trade_json.strip()

        # Debug Print (optional)
        Print(trade_json)

        # Parse JSON
        trade_json = json.loads(trade_json)

        time_taken = time.time() - start
        latency = get_latency_time(stream_mode)

        return time_taken, latency, trade_json

    except json.JSONDecodeError as e:
        # Save raw text to json_error.txt
        json_error_path = os.path.join(os.path.dirname(__file__), "json_error.txt")
        with open(json_error_path, "a", encoding="utf-8") as f:
            f.write(trade_json + "\n\n")
        Print(f"Error decoding JSON: {e}", log_path="errors.txt")
        return None

# def check_if_level_moving(trade_json,start,stream_mode,crop_screen,check_x_scale,logo_scaling):
#     new_frame = capture_screen(2)
#     if crop_screen:
#         new_frame,logo_exists,total_logos,matches = crop_chart_region(new_frame)
#     trades = scrape_screen(new_frame,check_x_scale,logo_scaling)
#     time_taken, latency, trade_json_2 = get_level_data(new_frame, trades, start, stream_mode)
#     if trade_json_2 == None:
#         return True,trade_json
#     sl_1 = trade_json.get("sl")
#     pair_1 = trade_json.get("pair")
#     tp_1 = trade_json.get("tp")
    
#     sl_2 = trade_json_2.get("sl")
#     pair_2 = trade_json_2.get("pair")
#     tp_2 = trade_json_2.get("tp")
#     if pair_1 != pair_2:
#         Print("pair has finished moving")
#         return True,trade_json
#     if sl_1 != sl_2 or tp_1 != tp_2:
#         Print("level is moving")
#         time.sleep(2)
#         new_frame = capture_screen(2)
#         trades = scrape_screen(new_frame,check_x_scale,logo_scaling)
#         time_taken, latency, trade_json = get_level_data(new_frame, trades, start, stream_mode)
#         if trade_json == None:
#             return True,trade_json_2
#         return True,trade_json
#     else:
#         return True,trade_json


def check_pair_in_only_pairs(trader_id, pair_name, trades_data, trader_config):
    if trader_config.get("only_pairs"):
        if pair_name not in trader_config["only_pairs"]:
            Print(f"[IGNORED] Pair {pair_name} not in allowed list for trader {trader_id}.")
            return False
    
    return True

def check_trader_in_ignore(trader_config,trader_id):
    if trader_config.get("ignore", False):
        Print(f"[IGNORED] Trader {trader_id} is ignored entirely.")
        # return trades_data
        return False
    else :
        return True



def process_trades(pair_name, frame, trades, trades_data, video_link, start, time_s, stream_mode, full_img, crop_screen, name, check_x_scale, scaling ,pair_img,is_micro,logo_size,or_frame):
    if not pair_name:
        Print("pair name is none")
        return trades_data

    # Determine trader ID
    if not name:
        """config"""
        trader_id = get_trader_name(full_img)
        if trader_id == "unknown":
            trader_id = "unk" + str(round(scaling, 4))
    else:
        trader_id = name

    current_time = time.time()
    Print("__trades__")
    Print(trades)

    # Ensure trader config exists
    if trader_id not in trades_data:
        trades_data[trader_id] = {
            "active": {},
            "unknown": {},
            "rejected": {},
            "config": {
                "ignore": False,
                "ignore_pairs": [],
                "only_pairs": [],
                "use_custom_risk": False,
                "custom_risk": None
            }
        }

    trader_config = trades_data[trader_id].get("config", {})
    trader_trades = trades_data[trader_id]["active"]
    unknown_signals = trades_data[trader_id]["unknown"]
    reject_trader_trades = trades_data[trader_id]["rejected"]

    # ===== Trader Ignore Check =====
    # if trader_config.get("ignore", False):
    #     Print(f"[IGNORED] Trader {trader_id} is ignored entirely.")
    #     return trades_data

    # # ===== Pair Ignore Check =====
    # if pair_name in trader_config.get("ignore_pairs", []):
    #     Print(f"[IGNORED] Pair {pair_name} ignored for trader {trader_id}.")
    #     return trades_data

    # ===== Pair Whitelist Check =====
    # if trader_config.get("only_pairs"):
    #     if pair_name not in trader_config["only_pairs"]:
    #         Print(f"[IGNORED] Pair {pair_name} not in allowed list for trader {trader_id}.")
    #         return trades_data

    # ===== Check if pair is in only_pairs =====
    if not check_pair_in_only_pairs(trader_id, pair_name, trades_data, trader_config):
        return trades_data
    
    if not check_trader_in_ignore(trader_config,trader_id):
        return trades_data

    if trades:
        # trades = trades[0]

        # ===== Handle Unknown Trades =====
        if trades["trade_type"] == "unknown" and trades["label"] != "just_tp":
            if pair_name in trader_trades and not in_sym_search(frame ,logo_size,name):
                trade_key = f"{pair_name}_{trader_trades[pair_name]['trade_type']}"
                unknown_signals[trade_key] = unknown_signals.get(trade_key, 0) + 1
                Print(f"[DEBUG] Unknown signal count for trader {trader_id}, pair {pair_name}: {unknown_signals[trade_key]}")

                if unknown_signals[trade_key] >= 1:
                    trade_data = trader_trades.pop(pair_name)
                    # close_trade(pair_name, pair_name, trader_id=trader_id, video=video_link)
                    Print(f"[CLOSED] {pair_name} ({trades['trade_type']}) closed at {current_time} because the number of unknown signals has reached 3.")
                    save_trade_event(pair_name, trader_id, "closed", current_time, time_s, video_link)
                    unknown_signals[trade_key] = 0

        else:
            # ===== Reset Unknown Signal Count =====
            if pair_name in trader_trades and trades["trade_type"] != "unknown":
                trade_key = f"{pair_name}_{trader_trades[pair_name]['trade_type']}"
                if trade_key in unknown_signals and unknown_signals[trade_key] != 0:
                    Print(f"[RESET] Resetting unknown signal count for trader {trader_id}, pair {pair_name}")
                    unknown_signals[trade_key] = 0

            # ===== New Trade =====
            if pair_name not in trader_trades and pair_name not in reject_trader_trades and not in_sym_search(frame ,logo_size,name) and trades["trade_type"] != "unknown":

                # Pass custom risk if enabled
                # if trader_config.get("use_custom_risk") and trader_config.get("custom_risk") is not None:
                #     open_trade(pair_name, trades["trade_type"], pair_name, video=video_link, trader_id=trader_id, risk=trader_config["custom_risk"])
                # else:
                #     open_trade(pair_name, trades["trade_type"], pair_name, video=video_link, trader_id=trader_id)

                # time_taken, latency, trade_json = get_level_data(frame, trades, start, stream_mode)
                # if trade_json == None:
                #     Print("Pair confirmation JSON is None ...closing trade")
                #     close_trade(pair_name, pair_name, video=video_link, trader_id=trader_id)
                #     Print(f"[CLOSED] {pair_name} closed at {current_time}")
                #     save_trade_event(pair_name, trader_id, "closed", current_time, time_s, video_link)

                # Validate Pair Name
                # if map_pairs(trade_json["pair"]) != map_pairs(pair_name):
                #     reject_trader_trades[pair_name] = trader_trades.pop(pair_name)
                #     Print(f"The name of the pair does not match. Local: {pair_name}, AI: {trade_json['pair']}")
                #     close_trade(pair_name, pair_name, video=video_link, trader_id=trader_id)
                #     Print(f"[CLOSED] {pair_name} closed at {current_time}")
                #     save_trade_event(pair_name, trader_id, "closed", current_time, time_s, video_link)

                Print("The name of the pair matches")
                trader_trades[pair_name] = {
                    "trade_type": trades["trade_type"],
                    "open_time": current_time,
                    "sl": trades["sl"],
                    "tp": trades["tp"],
                    "status": trades["status"]
                }
                # level_moved = False
                save_trade_event(pair_name, trader_id, "opened", current_time, time_s, video_link)

                # SL check
                if trades["sl"]:
                    # Print("checking if sl is moving")
                    # level_moved, trade_json = check_if_level_moving(trade_json, start, stream_mode, crop_screen, check_x_scale, logo_scaling=scaling)
                    # pip_risk = abs(trade_json["entry_price"] - trade_json["sl_price"])

                    # if trader_config.get("use_custom_risk") and trader_config.get("custom_risk") is not None:
                    #     recalculate_risk(time_taken, latency, pair_name, pip_risk, trade_json["pair"], video=video_link, trader_id=trader_id, risk=trader_config["custom_risk"])
                    # else:
                    #     recalculate_risk(time_taken, latency, pair_name, pip_risk, trade_json["pair"], video=video_link, trader_id=trader_id)
                    save_trade_event(pair_name, trader_id, "recalculate_sl", current_time, time_s, video_link)

                # TP check
                if trades["tp"]:
                    # Print("checking if tp is moving")
                    # if not level_moved:
                    #     level_moved, trade_json = check_if_level_moving(trade_json, start, stream_mode, crop_screen, check_x_scale, logo_scaling=scaling)
                    # update_trade(pair_name, "tp", trade_json["tp_price"], trade_json["pair"], video=video_link, trader_id=trader_id)
                    save_trade_event(pair_name, trader_id, "update_tp", current_time, time_s, video_link)

                Print(f"[OPENED] {pair_name} {trades} at {current_time}")
                

            # ===== Existing Trade Update =====
            elif pair_name in trader_trades:
                level_moved = False
                got_trades = False

                # if (trades["sl"] and not trader_trades[pair_name]["sl"]) or (trades["tp"] and not trader_trades[pair_name]["tp"]):
                #     time_taken, latency, trade_json = get_level_data(frame, trades, start, stream_mode)
                    
                #     if trade_json == None :
                #         pass
                #     got_trades = True

                if trades["sl"] and not trader_trades[pair_name]["sl"] and not in_sym_search(frame ,logo_size,name):
                    # Print("checking if sl is moving")
                    # if not got_trades:
                    #     time_taken, latency, trade_json = get_level_data(frame, trades, start, stream_mode)
                    
                    # if trade_json != None:
                    #     pip_risk = abs(trade_json["entry_price"] - trade_json["sl_price"])
                    trader_trades[pair_name]["sl"] = True

                    #     if trader_config.get("use_custom_risk") and trader_config.get("custom_risk") is not None:
                    #         recalculate_risk(time_taken, latency, pair_name, pip_risk, trade_json["pair"], video=video_link, trader_id=trader_id, risk=trader_config["custom_risk"])
                    #     else:
                    #         recalculate_risk(time_taken, latency, pair_name, pip_risk, trade_json["pair"], video=video_link, trader_id=trader_id)
                    # else :
                    #     Print("Skipping trade risk recalculation because trade_json is None",log_path="errors.txt")
                    save_trade_event(pair_name, trader_id, "recalculate_sl", current_time, time_s, video_link)
                if trades["tp"] and not trader_trades[pair_name]["tp"] and not in_sym_search(frame ,logo_size,name):
                    # Print("checking if tp is moving")
                    # if not got_trades:
                    #     time_taken, latency, trade_json = get_level_data(frame, trades, start, stream_mode)
                    # if trade_json != None:
                    #     if not level_moved:
                    #         level_moved, trade_json = check_if_level_moving(trade_json, start, stream_mode, crop_screen, check_x_scale, logo_scaling=scaling)
                    trader_trades[pair_name]["tp"] = True
                    #     update_trade(pair_name, "tp", trade_json["tp_price"], trade_json["pair"], video=video_link, trader_id=trader_id)
                    # else:
                    #     Print("Skipping trade TP update because trade_json is None",log_path="errors.txt")
                    save_trade_event(pair_name, trader_id, "update_tp", current_time, time_s, video_link)
                Print(f"[ONGOING] {pair_name} still open")

    else:
        # ===== No Trades — Close If Active =====
        if pair_name in trader_trades:
            trade_data = trader_trades.pop(pair_name)
            Print(trade_data)
            # close_trade(pair_name, pair_name, video=video_link, trader_id=trader_id)
            Print(f"[CLOSED] {pair_name} closed at {current_time}")
            save_trade_event(pair_name, trader_id, "closed", current_time, time_s, video_link)

    return trades_data


def is_paper_acc(matches,image):

    # Print("Matches:", matches)
    # x,y = matches[0]
    # x2,y2 = matches[0]
    # # _,width,height = matches[0][1] 
    # width = x2 - x
    # height = y2 - y
    
    # x_type = (width//3) + width + x
    # y_type = height//2 + y
    # blue_per = blue_percentage(image[y:y_type,x_type:x_type+width])
    # Print("The blue percentage is:",blue_per)
    # if blue_per != 0:
    #     if blue_per > 7:
    #         return True 
    return False

def process_1_screen(frame,check_x_scale,scaling,check_limit_orders,video_link,start,time_s,stream_mode,or_frame,crop_screen,name,trades_data,screen_data,logo_size):
    
    pair_name = screen_data["pair"]
    trades = screen_data["trades"]
    trades["label"] = screen_data["label"]
    if pair_name.startswith("M"):
        is_micro = True
    else :
        is_micro = False
    trades_data = process_trades(pair_name=pair_name,frame=frame,trades=trades,trades_data=trades_data,video_link=video_link,start=start,time_s=time_s,stream_mode=stream_mode,full_img=or_frame,crop_screen=crop_screen,name=name,check_x_scale=check_x_scale,scaling=scaling,pair_img=None,is_micro=is_micro,logo_size=logo_size,or_frame=or_frame)

    return trades_data


def create_or_append_number(filename: str, number: int):
    """
    Creates or overwrites the file so that it always contains exactly one number.
    """
    with open(filename, 'w') as f:  # 'w' clears file before writing
        f.write(str(number) + "\n")



def read_file(filename: str) -> Optional[int]:
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        if not lines:
            return None
        return int(lines[-1])

def crop_dee(frame, logo_size):
    frame_height, frame_width = frame.shape[:2]
    print("Logo")
    print(logo_size)
    face_h_size = (logo_size * 681) // 52
    face_w_size = (logo_size * 703) // 52

    y_start = max(0, face_h_size)
    x_end = min(face_w_size, frame_width)

    frame[y_start:frame_height, 0:x_end] = (0, 0, 0)
    
    return frame

def crop_frame(image ,logo_loc , logo_size, name):
    if name.lower() == "dee":
        print("cropping dee ")
        frame = crop_dee(image,logo_size)
    else:
        frame = image.copy()
    
    # from dump import display_image
    # display_image(frame,"balcked")
    # logo_loc = boxes[0]
    # logo_top_left , logo_bottom_right = logo_loc
    # logo_top_left_x , logo_top_left_y  = logo_top_left
    # image_h , image_w = image.shape[:2]
    # frame = image[logo_top_left_y:image_h , logo_top_left_x:image_w]
    frame = crop_right(frame,logo_loc,logo_size)        

    return frame 

# def crop_all_screen(image):
    
#     image_h , image_w = image.shape[:2]

    
#     x_start = (image_w * 285) // 1922
#     y_start = (image_h * 88) // 1080
#     y_end = (image_h * (1080 - 141)) // 1080
    
#     frame = image[y_start:y_end , x_start: image_w]
#     # from dump import display_image
#     # display_image(frame,"crop_all_screen")
#     # frame = image[88:939 , 285:image_w]
#     # frame = crop_right(frame,logo_loc,logo_size)
#     return frame 




def crop_all_screen(image):
    """
    screen 2 :
        x_start = 436
        y_start = 117
        y_end = 138

    screen 3 :
        x_start = 427
        y_start = 117
        y_end = 138
    """
    """
    screen 3 :
        x_start = 366
        y_start = 117
        y_end = 138
    """

    image_h , image_w = image.shape[:2]
    face_array = detect_faces(image)
    faces_normal = len(face_array)
    faces_text = detect_faces_text(image)
    faces = max(faces_normal,faces_text)
    print(f"There are {faces} faces")
    x_start = 366 
    if faces == 1 :
        x_start = 424
    if faces == 2 :
        x_start = 436
    elif faces == 3:
        x_start = 427
    elif faces == 4 :
        x_start = 366
    x_start = (image_w * x_start) // 1920
    y_start = (image_h * 117) // 1080
    y_end = (image_h * (1080 - 138)) // 1080
    
    frame = image[y_start:y_end , x_start: image_w]
    # from dump import display_image
    # display_image(frame,"crop_all_screen")
    # frame = image[88:939 , 285:image_w]
    # frame = crop_right(frame,logo_loc,logo_size)
    return frame 


def post_process_screens(screens):
    processed_screens = []
    seen_pairs = set()

    # Step 1: Pair up screens for the same asset
    for i in range(len(screens)):
        for j in range(i + 1, len(screens)):
            s1, s2 = screens[i], screens[j]
            s1["label"] = "None"
            s2["label"] = "None"
            if s1['pair'] == s2['pair'] and s1['pair'] not in seen_pairs:
                seen_pairs.add(s1['pair'])
                
                t1, t2 = s1['trades'], s2['trades']
                
                # CASE 1: One screen is missing data (unknown)
                if t1['trade_type'] != "unknown" and t2['trade_type'] == "unknown":
                    processed_screens.append(s1)
                elif t2['trade_type'] != "unknown" and t1['trade_type'] == "unknown":
                    processed_screens.append(s2)

                # CASE 2: Trade Types differ
                elif t1['trade_type'] == 'buy' and t1['status'] == 'profit' and t2['trade_type'] != 'buy' and t2['status'] != 'profit':
                    if t2["tp"]:
                        processed_screens.append(s2)
                    else:
                        s1["trade_type"] = "unknown"
                        s1["tp"] = True
                        s1["label"] = "just_tp"
                        processed_screens.append(s1)
                        processed_screens.append(s2)
                
                elif t1['trade_type'] != 'buy' and t1['status'] != 'profit' and t2['trade_type'] == 'buy' and t2['status'] == 'profit':
                    if t1["tp"]:
                        processed_screens.append(s1)
                    else:
                        s2["trade_type"] = "unknown"
                        s2["tp"] = True
                        s2["label"] = "just_tp"
                        processed_screens.append(s1)
                        processed_screens.append(s2)

                else:
                    processed_screens.append(s1)
                    processed_screens.append(s2)

    for s in screens:
        if s['pair'] not in seen_pairs:
            s['label'] = "None"
            processed_screens.append(s)

    return processed_screens

def process_frame(or_frame,time_s,video_link,trades_data,stream_mode,check_double_screen=True,crop_screen=False,name=False,check_x_scale=False,trader_does_not_have_logo=False,check_paper_acc=True,check_limit_orders=True):
        logo_exists = False
        logo_size = None
        logo_loc = None
        logo_locs = None
        matches = None
        total_logos = 0
        scaling = None
        # cv2.namedWindow("all frame", cv2.WINDOW_NORMAL)
        # cv2.imshow("all frame",or_frame)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # from dump import display_image
        # display_image(or_frame,"orig")

        frame = None
        try:        
            if name == "all":
                name = get_trader_name(or_frame)
                if name == "unknown":
                    return trades_data

            if or_frame is None or not isinstance(or_frame, np.ndarray):
                Print("Invalid frame received", log_path="errors.txt")
                return trades_data

            whiteness = is_main_color_white(or_frame)
            if  whiteness > 20:
                return trades_data

            print("Crop screen: ",crop_screen)
            if not crop_screen:
                frame = crop_all_screen(or_frame)
            else:
                frame = or_frame.copy()
            
            # from dump import display_image
            # display_image(frame, "cropped")
            logo_template = cv2.imread(os.path.join(os.path.dirname(__file__), "templates", "x_logo.png"))
            logo_size = None
            cached_logo_height = None
            print("The name in process frame is: ",name)
            print("The height in the config file is: ",get_config("best_logo_height",name= name))
            print("Type: ",type(get_config("best_logo_height",name= name)))
            if name == "Dakota" and get_config("is_alt",False,"Dakota"):
                name = "Jay_d" 
            if name == "Jay" and get_config("is_alt",False,"Dakota"):
                name = "Hoop_j" 
            if name == "Dakota" :
                cached_logo_height = get_config("best_logo_height",name= name)
            if name == "Jay_d":
                cached_logo_height = get_config("best_logo_height",name= name)
            if name == "Jay":
                cached_logo_height = get_config("best_logo_height",name= name)
            if name == "Hoop_j":
                cached_logo_height = get_config("best_logo_height",name= name)

            logo_size,_,logo_locs = detect_best_logo_height(frame,logo_template,name = name,trader_does_not_have_logo = trader_does_not_have_logo)
            # print("Updated height in the config file is: ",logo_size)
            if cached_logo_height != None:
                # print("cahched height is not none")
                # print("cached_logo_height: ",cached_logo_height)
                # print("logo size: ",logo_size)
                # print("name: ",name)
                if logo_size != cached_logo_height and name == "Dakota":
                    print("name updated to jayd")
                    name = "Jay_d"
                    update_config("is_alt",True,"Dakota")
                elif logo_size != cached_logo_height and name == "Jay_d":
                    name = "Dakota"
                    update_config("is_alt",False,"Dakota")
                elif logo_size != cached_logo_height and name == "Jay":
                    name = "Hoop_j"
                    update_config("is_alt",True,"Jay")
                elif logo_size != cached_logo_height and name == "Hoop_j":
                    name = "Jay"
                    update_config("is_alt",False,"Jay")
                else:
                    pass
            
            print("Revised name is: ",name)

            if not logo_locs:  # empty or None
                logo_exists = False
                logo_size = None
                logo_loc = None
                matches = None
                total_logos = 0
            else:
                logo_exists = True
                logo_loc = logo_locs[0]
                matches = logo_locs[0]
                total_logos = len(logo_locs)
                scaling = logo_size
            
            if not crop_screen:
                frame = crop_right(frame,logo_loc,logo_size)
            else:
                if logo_exists:
                    print("logo does not exist")
                    frame = crop_frame(or_frame, logo_loc, logo_size,name)
                else:
                    Print("Cannot crop frame: logo info missing", log_path="errors.txt")
                    frame = or_frame.copy()
            # else:
            #     frame = or_frame.copy()
            if frame is None:
                frame = or_frame.copy()

                # print("not cropping crop_screen is False")
            # if in_sym_search(frame ,logo_size,name):
            #     return trades_data
                      
            if logo_exists:
                if total_logos > 1:
                    Print("There is more than one logo",log_path="errors.txt")
                    return trades_data
                if logo_exists and check_paper_acc and not trader_does_not_have_logo:
                    print("matches")
                    print(matches)
                    if matches and is_paper_acc(matches, or_frame):
                        return trades_data

                start = time.time()
                
                # display_image(frame,"final_processed_image")
                screens_data = get_data(frame,logo_size,logo_loc,name)
                print(screens_data)
                screens_data = post_process_screens(screens_data)
                print("POST PROCESSING OF SCREENS DATA")
                print(screens_data)
                
                if screens_data:
                    for screen_data in screens_data:
                        trades_data = process_1_screen(frame=frame,check_x_scale=check_x_scale,scaling=scaling,video_link=video_link,start=start,time_s=time_s,check_limit_orders=check_limit_orders,stream_mode=stream_mode,or_frame=or_frame,crop_screen=crop_screen,name=name,trades_data=trades_data,screen_data=screen_data,logo_size=logo_size)
                            
            else:
                Print("No logo found")
                return trades_data
                    
            return trades_data

        except Exception as e:
            Print(f"[ERROR in main] {e}",log_path = os.path.join(os.path.dirname(__file__),"errors.txt"))
            log_exception(log_path = os.path.join(os.path.dirname(__file__),"errors.txt"))
            return trades_data


if __name__ == "__main__":
    img = cv2.imread("screen.png")
    logo_scaling = 1
    # check_limit_orders = True
    # if check_limit_orders:
    #     has_limit_order = any(
    #         trade.get("trade_type") == "limit"
    #         for trade in trades
    #     )
        
    #     if has_limit_order:
    #         Print("There is a limit order")
    # Print("trades")
    