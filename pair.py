''''
1. get the boxes from the image 
2. look for matches with the saved templates using cv2
3. If there are no matches with over 95% match 
4. Use the fnctis 
5. Use the paddle_inf function to get the new matches
6. Run the results though name.py function
7. If there is a match crop the image and save it in the folder pair_templates under the trade_name sub_folder

'''
import cv2
import os
from easy_boxes import get_boxes
from paddle_inf import get_text_detection_model,recognize_text
import numpy as np
# from paddle_inf import run_ocr
# from name import get_pairs 
import random
import string
# from ad_pairs import match_futures_symbol
from font_dim import get_text_range
# from easy_boxes import get_boxes
"""
This is the script for advanced text filtering and matching of the search futures pairs from paddle OCR correctly :
This is a list of characters that confuse OCR's :
A
B = 8
C = G = 6
D = 0 = O = Q
E = F
H
I = J = L = T = 1 = 7
K = X
M
N
P
R
S = 5
U = V = Y
W
Z = 2
3
4
6
9

The next thing we need need to do is remove everything after a non-alphanumeric character
The last two values must be numeric for the year
The third last value must be one of these month codes : F (January), G (February), H (March), J (April), K (May), M (June), N (July), Q (August), U (September), V (October), X (November), and Z (December)
NOTE The month and year are not a priority since the goal is to map the futures pair
micro_map = { "MNQ", "MES", "MCL", "MGC", "MBT", "MHG", "MNG", "FDX","SIL"}
base_map = { "ES", "NQ", "YM", "CL", "GC", "HG", "NG", "PL", "RB", "ZB", "ZF", "6J", "6B","NKD","RTY","CPE","SI"}

Here is a list of the function steps 
1. Clean the text
- Remove non alphanumeric characters  
- Any other form of cleaning you think is necessary
2. Check if the text starts with any of the base symbols , if yes then that is a match (direct_match)
3. If we did not get a direct match ,use the list of ambiguos characters looping to trying to find a direct match
4. If we still do not get a match return the best match that has an equal number amount of characters but the OCR only missed one character eg ANQZ25 would be a match for MNQ but not NQ

"""


import re
import logging
from itertools import product

# ----------------------------
# LOGGING SETUP
# ----------------------------
logger = logging.getLogger("futures_symbol_matcher")
logger.setLevel(logging.DEBUG)   # Change to DEBUG to reduce noise

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Format
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


# ----------------------------
# CONFIGURATION
# ----------------------------

OCR_GROUPS = [
    {"A"},
    {"B", "8"},
    {"C", "G", "6"},
    {"D", "0", "O", "Q"},
    {"E", "F"},
    {"H"},
    {"I", "J", "L", "T", "1", "7"},
    {"K", "X"},
    {"M"},
    {"N"},
    {"P"},
    {"R"},
    {"S", "5"},
    {"U", "V", "Y"},
    {"W"},
    {"Z", "2"},
    {"3"},
    {"4"},
    {"6"},
    {"9"},
]

DIGITS      = set("0123456789")
ALPHABETS   = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ----------------------------
# SYMBOL MAPS
# ----------------------------
MICRO_MAP = {"MNQ", "MES", "MCL", "MGC", "MBT", "MHG", "MNG", "FDX", "SIL"}
BASE_MAP  = {"ES", "NQ", "YM", "CL", "GC", "HG", "NG", "PL", "RB",
             "ZB", "ZF", "NKD", "RTY", "CPE", "SI"}

ALL_SYMBOLS = MICRO_MAP | BASE_MAP

MONTH_CODES = set("FGHJKMNQUVXZ")



# ----------------------------
# OCR GROUP LOOKUP
# ----------------------------

def build_char_to_group_map(groups):
    char_to_group = {}
    for idx, group in enumerate(groups):
        for char in group:
            char_to_group[char] = idx
    return char_to_group


CHAR_TO_GROUP = build_char_to_group_map(OCR_GROUPS)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def get_canonical_match(char, allowed_set, groups, char_to_group):
    """
    Resolve a character into a canonical form allowed by `allowed_set`
    using direct match or fuzzy OCR group matching.
    """
    if not char:
        return None

    char = char.upper()

    # 1. Direct match
    if char in allowed_set:
        return char

    # 2. Fuzzy OCR group match
    if char in char_to_group:
        group_idx = char_to_group[char]
        group = groups[group_idx]
        matches = group.intersection(allowed_set)
        if matches:
            return list(matches)[0]

    return None

# ----------------------------
# MAIN FUNCTION
# ----------------------------

def confirm_future_pair_nos(ocr_text):
    if not ocr_text:
        return None

    text = ocr_text.upper().strip()

    YEAR_LEN = 2  # 🔒 exactly 2 characters

    # Minimum length: 2-char symbol + 1 month + 2-digit year = 5
    if len(text) < 5:
        return None

    # Split text
    raw_year   = text[-YEAR_LEN:]
    raw_month  = text[-(YEAR_LEN + 1)]
    raw_symbol = text[:-(YEAR_LEN + 1)]

    # 🔒 Symbol length constraint
    if not (2 <= len(raw_symbol) <= 3):
        return None

    # --- YEAR ---
    corrected_year = ""
    for char in raw_year:
        digit = get_canonical_match(char, DIGITS, OCR_GROUPS, CHAR_TO_GROUP)
        if not digit:
            return None
        corrected_year += digit

    # --- MONTH ---
    corrected_month = get_canonical_match(
        raw_month, MONTH_CODES, OCR_GROUPS, CHAR_TO_GROUP
    )
    if not corrected_month:
        return None

    # --- SYMBOL ---
    corrected_symbol = ""
    for char in raw_symbol:
        letter = get_canonical_match(
            char, ALPHABETS, OCR_GROUPS, CHAR_TO_GROUP
        )
        if not letter:
            return None
        corrected_symbol += letter

    return f"{corrected_symbol}{corrected_month}{corrected_year}"


# ----------------------------
# CLEAN TEXT
# ----------------------------
def clean_text(text: str , search: bool = False) -> str:
    logger.debug(f"Raw OCR text received: {text!r}")

    text = text.upper().strip()
    logger.debug(f"Uppercased: {text!r}")
    if not search and (text.startswith("/") or text.startswith("I")):
        search = True
    if search and text.startswith("/"):
            text = text[1:]
            if "1" in text:
                    text = text.replace("1", "I")
    if search and text.startswith("I"):
            text = text[1:]   
            if "1" in text:
                    text = text.replace("1", "I")

    match = re.match(r"[A-Z0-9]+", text)

    if not match and not search :
        logger.warning("No alphanumeric prefix found in OCR text.")
        return None, None

    if not match and search:
        cleaned = match

    else:
        cleaned = match.group(0)



        
    logger.info(f"Cleaned text extracted: {cleaned}")
    return cleaned,search


# ----------------------------
# DIRECT MATCH
# ----------------------------
def direct_match(symbol: str):
    logger.debug(f"Attempting direct match for {symbol!r}")
    
    for s in MICRO_MAP:
        if symbol.startswith(s):
            logger.info(f"Direct match found: {s}")
            return s , True 
    for s in BASE_MAP:
        if symbol.startswith(s):
            logger.info(f"Direct match found: {s}")
            return s , False
            
    logger.debug("No direct match.")
    return None,None


# ----------------------------
# AMBIGUITY HANDLING
# ----------------------------
def get_ambiguity_candidates(char: str):
    for group in OCR_GROUPS:
        if char in group:
            return list(group)
    return [char]


def try_ambiguity_substitutions(symbol: str):
    logger.debug(f"Trying ambiguity substitutions for: {symbol}")

    max_len = max(len(s) for s in ALL_SYMBOLS)
    prefix = symbol[:max_len]

    candidate_lists = [get_ambiguity_candidates(c) for c in prefix]

    logger.debug(f"Candidate lists: {candidate_lists}")

    for combo in product(*candidate_lists):
        attempt = "".join(combo)
        match,is_micro = direct_match(attempt)
        if match:
            logger.info(f"Ambiguous substitution matched: {attempt} → {match}")
            return match,is_micro

    logger.debug("No ambiguity-based match found.")
    return None,None


# ----------------------------
# SINGLE-CHAR FALLBACK
# ----------------------------
def best_single_char_match(symbol: str, search: bool = False):
    """
    Step 4 (Updated):
       - Remove final 3 chars (month + year)
       - Only compare the base portion to all known futures symbols
       - Match only if there is exactly 1 character mismatch
    """
    logger.debug(f"Checking for single-character fuzzy matches for: {symbol}")

    # --- NEW: Trim month+year ---
    if not search:
        if len(symbol) > 3:
            base_candidate = symbol[:-3]
            logger.debug(f"Trimmed symbol for fuzzy match: {base_candidate}")
        else:
            base_candidate = symbol
            logger.debug(f"Symbol too short to trim, using as-is: {base_candidate}")

    else:

        base_candidate = symbol
    length = len(base_candidate)
    best = None
    is_micro = False
    for s in ALL_SYMBOLS:
        if s in MICRO_MAP:
            is_micro = True
        if len(s) != length:
            continue

        mismatches = sum(1 for a, b in zip(base_candidate, s) if a != b)

        if mismatches == 1:
            logger.info(f"Single-char fuzzy match: {base_candidate} → {s}")
            best = s
            break

    if not best:
        logger.debug("No single-character fuzzy matches found.")

    return best,is_micro

# ----------------------------
# MAIN ENTRY FUNCTION
# ----------------------------

def alphanum(s: str) -> bool:
    """
    Checks if a string contains at least one alphabetic character and 
    at least one numeric character using built-in string methods.
    
    Args:
        s: The input string.
        
    Returns:
        True if the string contains both letters and numbers, False otherwise.
    """
    if not s:
        return False

    has_alpha = any(c.isalpha() for c in s)
    has_digit = any(c.isdigit() for c in s)

    return has_alpha and has_digit

def match_futures_symbol(text: str, search: bool = False):
    logger.info("===== START MATCHING PROCESS =====")
    
    cleaned,search = clean_text(text,search=search)
    print("search : ",search)
    if not cleaned:
        logger.error("Could not clean text into usable symbol.")
        return None    
    if not search:
        print("We are not searching")
        is_alphanum = alphanum(cleaned)
        print("isalphanum: ",is_alphanum)
        if not is_alphanum:
            return None
        if confirm_future_pair_nos(cleaned) == None:
            return None
    else:
        is_alphanum = alphanum(cleaned)
        if is_alphanum:
            return None
        if len(cleaned) > 4:
            return None

    raw = text 
    extracted = cleaned 
    


    # Step 2: Direct match
    match,is_micro = direct_match(cleaned)
    
    if match:
        mapped = match
        fuzzy = False
        symbol_data = {"raw":raw ,
                       "extracted":extracted,
                       "mapped": mapped,
                       "fuzzy":fuzzy,
                       "is_micro":is_micro} 
        
        logger.info(f"FINAL MATCH (direct): {match}")
        return symbol_data

    # Step 3: Ambiguous substitution
    match,is_micro = try_ambiguity_substitutions(cleaned)
    if match:
        mapped = match 
        fuzzy = True
        symbol_data = {
            "raw": raw ,
            "extracted": extracted,
            "mapped": mapped,
            "fuzzy": fuzzy,
            "is_micro": is_micro
        }
        logger.info(f"FINAL MATCH (ambiguity): {match}")
        return symbol_data

    # Step 4: Single character fallback
    match,is_micro = best_single_char_match(cleaned,search)
    if match:
        logger.info(f"FINAL MATCH (fuzzy): {match}")
        symbol_data = {
            "raw": raw ,
            "extracted": extracted,
            "mapped": match,
            "fuzzy": True,
            "is_micro": is_micro
        }
        return symbol_data

    logger.warning("No match found for symbol.")
    return None




# [[[np.int32(520), np.int32(544), np.int32(4), np.int32(10)], [np.int32(33), np.int32(81), np.int32(5), np.int32(17)], [np.int32(102), np.int32(228), np.int32(6), np.int32(14)], [np.int32(232), np.int32(276), np.int32(6), np.int32(14)], [np.int32(554), np.int32(584), np.int32(6), np.int32(14)], [np.int32(632), np.int32(804), np.int32(6), np.int32(14)]]]


def generate_random_strings(length):
    return ''.join(random.choice(string.ascii_letters ) for _ in range(length))

# def get_boxes(image):
#     detection_model = get_text_detection_model()
#     out_put = detection_model.predict(image,batch_size = 1) 
#     # print(out_put)
#     for out_puts in out_put:
#         out_puts.save_to_json(save_path = "lq_json.json")
#     polys = out_put[0].get("dt_polys")
#     rectangles = []
#     for poly in polys:
#         min_x = 9999
#         max_x = 0
#         min_y = 9999
#         max_y = 0
#         for point in poly:
#             x = point[0]
#             y = point[1]
#             if x < min_x :
#                 min_x = x
#             if x > max_x :
#                 max_x = x
#             if y < min_y :
#                 min_y = y
#             if y > max_y :
#                 max_y = y
        
#         rectangle = ((min_x,min_y),(max_x,max_y))
#         rectangles.append(rectangle)
    
#     # print("rectangles")
#     # print(rectangles)
#     return rectangles


def sanitize_filename(name):
    # Replace every character that is not A-Z, a-z, 0-9, or _ with _
    return re.sub(r'[^A-Za-z0-9_]', '_', name)



def get_easy_boxes(image):
    h, f = get_boxes(image=image)

    H, W = image.shape[:2]         # image height & width
    rectangles = []

    for box in h[0]:
        x1, x2, y1, y2 = box

        # Normalize order
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        # Clip to image boundaries
        x_min = max(0, min(x_min, W - 1))
        x_max = max(0, min(x_max, W - 1))
        y_min = max(0, min(y_min, H - 1))
        y_max = max(0, min(y_max, H - 1))

        rectangles.append(((x_min, y_min), (x_max, y_max)))

    return rectangles
    

def get_trash_name(trader,trash_folder):
    if not os.path.exists(os.path.join("pair_templates",trader,trash_folder)):
        os.mkdir(os.path.join("pair_templates",trader,trash_folder))
    
    trash_paths = os.listdir(os.path.join("pair_templates",trader,trash_folder))

    for i in range(0,10000):
        if f"trash_{i}.png" in trash_paths:
            continue
        else:
            return f"trash_{i}.png"



def get_pairs(image,search= False , trader = "dakota",logo_height = 44 , orig_y = 0):
# for pt in ["lq.png","dakota_2.png","dakota_3.png"]:
    # image = cv2.imread(pt)
    # cv2.imwrite("image_strip.png",image)
    # cv2.imshow("pairs_image",image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    if search:
        mini_folder = "search"
        trash_folder = "search_trash"
    else:
        mini_folder = "screen"
        trash_folder = "screen_trash"
    # print("The mini folder is: ",mini_folder)

    horizontal_boxes = get_easy_boxes(image)
    # print(horizontal_boxes)
    # trader = "dakota"
    # print(f"The trader in get_pairs is: {trader}")
    if not os.path.exists(os.path.join("pair_templates")):
        os.mkdir(os.path.join("pair_templates"))
    if not os.path.exists(os.path.join("pair_templates",trader)):
        os.mkdir(os.path.join("pair_templates",trader))


    if not os.path.exists(os.path.join("pair_templates",trader,mini_folder)):
        os.mkdir(os.path.join("pair_templates",trader,mini_folder))

    if not os.path.exists(os.path.join("pair_templates",trader,trash_folder)):
        os.mkdir(os.path.join("pair_templates",trader,trash_folder))

    pair_paths = os.listdir(os.path.join("pair_templates",trader,mini_folder))
    trash_paths = os.listdir(os.path.join("pair_templates",trader,trash_folder))
    
    # if not os
    print(f"There are {len(horizontal_boxes)} boxes")
    # print(horizontal_boxes)
    pair_found = False
    results = []
    non_trash_boxes = []
    # print(horizontal_boxes)
    text_range,text_height = get_text_range(logo_height)
    min_text_range = text_range[0]
    max_text_range = text_range[1]+5
    # print("min_text_range: ", min_text_range)
    # print("max_text_range: ", max_text_range)
    # print("Text height: ",text_height)
    for box in horizontal_boxes:

        x1,y1 = box[0]
        x2,y2 = box[1]
        # cv2.rectangle(image,box[0],box[1],(0,255,0),2)
        image_h , image_w = image.shape[:2]
        cropped = image[max(0,y1-3):min(image_h,y2+3) , max(0,x1-5):min(image_w,x2+5)]


        
        satisfies_range = False
        # print("Box range: ",x2 - x1)
        # print((max_text_range > (x2 - x1) > min_text_range ))
        # print("Box height: ",abs(y2 - y1))
        # print((abs(y2 - y1) > text_height+2))
        # for i in range(0,200):
            # print("The fuck is happenning")
        if (max_text_range > (x2 - x1) > min_text_range ) and (abs(y2 - y1) > text_height+2):
            satisfies_range = True
        if search :
            satisfies_range = True
        
        # print("Cropped satisfies range: ",satisfies_range)
        # from dump import display_image
        # display_image(cropped,"cropped")
        
        if satisfies_range:
            # cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),1)
            image_h , image_w = image.shape[:2]
            cropped = image[max(0,y1-3):min(image_h,y2+3) , max(0,x1-5):min(image_w,x2+5)]
            
            if len(trash_paths) != 0:
                is_not_trash = True
                for trash in trash_paths:
                    trash_path = os.path.join("pair_templates", trader, trash_folder , trash)
                    trash_template = cv2.imread(trash_path)
                    if trash_template is None:
                        # print(f"Warning: Could not load trash template {template_path}")
                        continue

                    # Skip if template is larger than the cropped image
                    if (trash_template.shape[0] > cropped.shape[0]) or (trash_template.shape[1] > cropped.shape[1]):
                        # print(f"Skipping {trash}: trash template larger than cropped image")
                        # non_trash_boxes.append(box)
                        continue

                    result = cv2.matchTemplate(cropped, trash_template, cv2.TM_CCOEFF_NORMED)
                    cropped_height , cropped_width = cropped.shape[:2]
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    x_1,y_1 = max_loc
                    temp_h , temp_w = trash_template.shape[:2]
                    x_2 = x_1 + temp_w   
                    y_2 = y_1 + temp_h   
                    # print("Match value:", max_val)
                    # print("X1: ",x_1)
                    # print("Cropped_width // 3: ", cropped_width // 3)
                    if max_val > 0.95 and x_1 < cropped_width // 3:
                        print("Trash found:", trash)
                        is_not_trash = False
                        break
                    else:
                        # print("This is not trash")
                        pass
                if is_not_trash:
                    non_trash_boxes.append(box)
                
            else:
                non_trash_boxes.append(box)
            # from dump import display_image            
            # display_image(cropped)

    # from dump import display_image
    # display_image(image,"tf")
    # print(f"There are {len(non_trash_boxes)} non trash boxes")
    unprocessed_boxes = []
    saved_pairs = set()

    for j, box in enumerate(non_trash_boxes):
        x1, y1 = box[0]
        x2, y2 = box[1]
        h,w = image.shape[:2]
        # print(f"Height: {h} , Width: {w}")
        # print(f"box 1 coords: x: {x1} , y: {y1}")
        # print(f"box 2 coords: x: {x2} , y: {y2}")
        cropped = image[y1:y2, x1:x2]
        # display_image(cropped,"nontrash")
        # cv2.imshow("c",cropped)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # cv2.rectangle(image ,box[0],box[1],(0,255,0),2)
        best_val = -1
        best_pair = None
        best_box = None

        for pair in pair_paths:
            template_path = os.path.join("pair_templates", trader, mini_folder, pair)
            template = cv2.imread(template_path)

            if template is None:
                print(f"Warning: Could not load template {template_path}")
                continue

            # Skip if template is larger than the cropped image
            if (template.shape[0] > cropped.shape[0]) or (template.shape[1] > cropped.shape[1]):
                print(f"Skipping {pair}: t;emplate larger than cropped image")
                continue

            result = cv2.matchTemplate(cropped, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            # print("Match value:", max_val)

            if max_val > best_val:  # track highest match
                x_1, y_1 = max_loc
                temp_h, temp_w = template.shape[:2]
                x_2 = x_1 + temp_w
                y_2 = y_1 + temp_h

                best_val = max_val
                best_pair = pair
                best_box = [
                    (x_1 + x1, y_1 + y1),
                    (x_1 + x1, y_2 + y1),
                    (x1 + x_2, y1 + y_2),
                    (x1 + x_2, y_1 + y1)
                ]

        # After checking all pairs

        if best_val >= 0.95:
            print("Best pair found:", best_pair, "Value:", best_val)
            print("The best box is: ")
            print(box)
            results.append((best_box, best_pair.replace(".png", ""), np.float64(best_val)))
            pair_name_clean = best_pair.replace(".png","")
            saved_pairs.add(pair_name_clean)
        else:
            print("No best match found , Highest match was: " ,best_pair, "Value:", best_val)
            unprocessed_boxes.append(box)

    # print(f"There are {len(unprocessed_boxes)} unprocessed boxes")
    # cv2.imshow("full_image",image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
        
    if len(unprocessed_boxes) != 0:
        # print("processing unprocessed boxes")
        for i,box in enumerate(unprocessed_boxes):
            x1,y1 = box[0]
            x2,y2 = box[1]
            cropped = image[y1:y2,x1:x2]
            text,score = recognize_text(cropped)
            top_left = box[0]
            bottom_right = box[1]
            top_left_x = top_left[0]
            top_left_y = top_left[1]
            bottom_right_x = bottom_right[0]
            bottom_right_y = bottom_right[1]

            converted = [(top_left_x,top_left_y),(top_left_x,bottom_right_y),(bottom_right_x,bottom_right_y),(bottom_right_x,top_left_y)]
            result = (converted,text,np.float64(score))
            results.append(result)

        
    # print("results",results)
    symbols_data = []
    for result in results:
        symbol_data = match_futures_symbol(result[1],search=search)
        if symbol_data is not None:
            symbols_data.append(symbol_data)
        else:
            box = result[0]
            top_left = box[0]
            bottom_right = box[2]
            top_left_x = top_left[0]
            top_left_y = top_left[1]
            bottom_right_x = bottom_right[0]
            bottom_right_y = bottom_right[1]
            cropped_trash = image[top_left_y:bottom_right_y , top_left_x:bottom_right_x]
            print("trader",trader)
            print("trash name,", get_trash_name(trader,trash_folder))
            save_trash_path = os.path.join("pair_templates",trader,trash_folder,get_trash_name(trader,trash_folder))
            print(save_trash_path)
            cv2.imwrite(save_trash_path,cropped_trash)
            """
                if a symbol is not matchable, crop it and save it in the trash folder 
            """
            # pass
    print("SYMBOLS DATA")
    print(symbols_data)
    print("RESULTS")
    print(results)
    used_results = []
    for symbol_data in symbols_data:
        for result in results:
            text = result[1]
            box = result[0]
            if result not in used_results:
                if text == symbol_data["raw"]:
                    used_results.append(result)
                    xs = [p[0] for p in box]
                    valid_x_start = int(min(xs))
                    symbol_data["x_start"] = valid_x_start
                    break



    if symbols_data:
        # print(symbols_data)
        for symbol in symbols_data:
            raw_text = symbol["raw"]
            
            # ---- SKIP if already saved ----
            if raw_text in saved_pairs:
                # print(f"Skipping {raw_text}, already saved earlier")
                continue
            # --------------------------------

            extracted_text = symbol["extracted"]
            mapped_text = symbol["mapped"]

            for result in results:
                text = result[1]
                if text == raw_text:
                    box = result[0]
                    top_left , bottom_left , bottom_right , top_right = box
                    top_left_x = top_left[0]
                    top_left_y = top_left[1]
                    bottom_right_x = bottom_right[0]
                    bottom_right_y = bottom_right[1]
                    cropped = image[top_left_y:bottom_right_y , top_left_x:bottom_right_x]

                    if "/" in raw_text:
                        raw_text = raw_text.replace("/", "")
                    clean_name = sanitize_filename(raw_text.replace(" ", "").replace(".", ""))

                    folder_path = os.path.join("pair_templates", trader, mini_folder)
                    os.makedirs(folder_path, exist_ok=True)

                    filename = f"{clean_name}.png"
                    save_path = os.path.join(folder_path, filename)
                    counter = 1

                    while os.path.exists(save_path):
                        filename = f"{clean_name}_{counter}.png"
                        save_path = os.path.join(folder_path, filename)
                        counter += 1

                    cv2.imwrite(save_path, cropped)
                    # print(f"Saved as: {filename}")
                    break

    
    if len(symbols_data) == 0 and search:
        symbols_data.append({})
    print(symbols_data)
    return symbols_data if not search else symbols_data[0]



if __name__ == "__main__":
    # confirmed_pair = confirm_future_pair_nos("C2541")
    # print(confirmed_pair)
    print(match_futures_symbol("/HG·5",False))
    print(match_futures_symbol("/HG·5",True))