import cv2
import os
import numpy as np
import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr
import cv2
import traceback
import re

from std_out import Print
from config import get_config,update_config
from sym import map_futures_symbol
from pair import get_easy_boxes
from pair import recognize_text


reader = None
def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=True)
    return reader
# reader.
# reader = None
BASE_SYMBOLS = ["ES","NQ","YM","CL","GC","HG","NG","PL","RB","ZB","ZF","6J","6B","NKD","RTY","CPE","MNQ","MES","MCL","MGC","MBT","MHG","MNG","FDXM","SI","SIL"]

def ocr(image):
    reader = get_reader()
    results = reader.readtext(image)
    texts = []
    print(results)
    for box, text, _ in results:
        texts.append(text)
    
    return texts

# if __name__ == "__main__":
#     cont = ( text_box,'MNQZ25', np.float64(0.8470726135202875))
#     text_box = cont[0]
#     text = cont[1]
#     image = cv2.imread("Screenshot 2025-11-29 200732.png")   
#     top_left = text_box[0]
#     bottom_left = text_box[1]
#     bottom_right = text_box[2]
#     top_right = text_box[3]
#     cv2.rectangle(image ,top_left,bottom_right,(0,255,0),2)
#     cv2.imshow("",image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def pick_text(results):
    alphanum_candidate = None
    alpha_candidate = None
    last_text = None

    for box, text, _ in results:
        last_text = text
        last_box = box

        # Check alphanumeric (letters+digits)
        if text.isalnum():
            alphanum_candidate = text
            alphanum_box = box
            break

        # Check alphabetic
        if text.isalpha():
            alpha_candidate = text
            alpha_box = box 

    if alphanum_candidate is not None:
        return alphanum_candidate,alphanum_box
    if alpha_candidate is not None:
        return alpha_candidate,alpha_box

    return last_text,last_box

MONTH_CODES = set("FGHJKMNQUVXZ")

def extract_month_year(symbol: str):
    if len(symbol) < 4:
        return None, None

    month = symbol[-3]
    year = symbol[-2:]

    if month in MONTH_CODES and year.isdigit():
        return month, year

    return None, None


def fuzzy_one_mismatch(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False

    mismatches = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            if ca.isalpha() and cb.isalpha():
                mismatches += 1
                if mismatches > 1:
                    return False
            else:
                return False
    return True


def fuzzy_match_with_month_year_arr(ocr_texts, month, year):
    if not month or not year:
        return None
    fuzzy_matches = []
    matched_clean_strings = []

    for o,box in ocr_texts:
        o_clean = o.replace(" ", "").upper()

        if not any(c.isalpha() for c in o_clean):
            continue

        for base in BASE_SYMBOLS:
            expected = f"{base}{month}{year}"
            if o_clean not in matched_clean_strings:
                if fuzzy_one_mismatch(o_clean, expected):
                    matched_clean_strings.append(o_clean)
                    
                    fuzzy_matches.append((expected,box))  # Found a match

    return fuzzy_matches
def fuzzy_match_with_month_year(ocr_texts, month, year):
    if not month or not year:
        return None
    fuzzy_matches = []

    for o,box in ocr_texts:
        o_clean = o.replace(" ", "").upper()

        if not any(c.isalpha() for c in o_clean):
            continue

        for base in BASE_SYMBOLS:
            expected = f"{base}{month}{year}"

            if fuzzy_one_mismatch(o_clean, expected):
                
                return expected,box  # Found a match

    return None,None


    


def match_template_with_best(image, template_path, threshold=0.95):
    """Original single-size match."""
    if not os.path.exists(template_path):
        return None, 0.0

    template = cv2.imread(template_path)
    if template is None:
        return None, 0.0

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return (max_loc, max_val) if max_val > threshold else (None, max_val)

def match_template_with_best_resized(image, template_path,scale, threshold=0.85):
    """Original single-size match."""
    if not os.path.exists(template_path):
        return None, 0.0

    template = cv2.imread(template_path)
    if template is None:
        return None, 0.0
    new_w = int(template.shape[1] * scale)
    new_h = int(template.shape[0] * scale)
    resized_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return (max_loc, max_val) if max_val > threshold else (None, max_val)


def match_template_resized(image, template_path, threshold=0.85):
    """Slower method: resize template from 0.65 to 1.75 scale."""
    template = cv2.imread(template_path)
    if template is None:
        return None, 0.0

    for scale in np.linspace(0.65, 1.75, num=12):
        if scale == 1.0:
            continue

        new_w = int(template.shape[1] * scale)
        new_h = int(template.shape[0] * scale)
        if new_w < 5 or new_h < 5:
            continue

        resized_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > threshold:
            return max_loc, max_val,scale

    return None, 0.0, 0.0


# === Unified worker functions ===
def process_original(name, cropped, threshold):
    """Try matching at original size."""
    
    template_path = os.path.join(os.path.join(os.path.dirname(__file__), "names"), name)
    match, val = match_template_with_best(cropped, template_path, threshold)
    if match is not None:
        return name, val, None  # scale=None for original
    return None


def process_resized(name, cropped, threshold):
    """Try matching with resizing."""
    template_path = os.path.join(os.path.join(os.path.dirname(__file__), "names"), name)
    match, val, scale = match_template_resized(cropped, template_path, threshold)
    if match is not None:
        return name, val, scale
    return None


def get_ocr_name(image):
    print("Tring ocr method for the name")
    # 1. Load images
    x_logo = cv2.imread("templates/x_logo.png")
    # image = cv2.imread("frame.png")
    
    if x_logo is None or image is None:
        print("Error: Could not load images.")
        return

    h, w = image.shape[:2]
    # 2. Crop the area where you expect the logo (Top Right)
    cropped_image = image[:h//3 , 3*(w//4):].copy() 
    from dump import display_image
    # display_image(cropped_image,"ocr")
    
    from resize import resize_proportional # Ideally move to top
    
    curr_max = 0
    best_match = None
    

    # 3. Iterate through possible sizes
    for height in range(5, 40):
        resized_template = resize_proportional(x_logo, height=height)
        
        # MATCH AGAINST cropped_image, NOT the full image
        result = cv2.matchTemplate(cropped_image, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > curr_max:
            curr_max = max_val
            t_h, t_w = resized_template.shape[:2]
            x, y = max_loc
            best_match = (max_loc, (x + t_w, y + t_h))
            best_height = height

    # 4. Draw and Display
    # if best_match and curr_max > 0.7: # Added a threshold check
        # cv2.rectangle(cropped_image, best_match[0], best_match[1], (0, 255, 0), 2)
    
    # display_image(cropped_image,"marked ocr")
    
    x,y = best_match[0]
    x2,y2 = best_match[1]
    
    cropped_part = cropped_image[y-5:y2+5 , :x]
    # display_image(cropped_part,"cro")
    text_boxes = get_easy_boxes(cropped_part)
    print(f"There are {len(text_boxes)} text boxes")
    non_trash_boxes = []
    for text_box in text_boxes:
        ((x_min, y_min), (x_max, y_max)) = text_box
        if y_max-y_min < best_height - 10:
            continue
        else:
            non_trash_boxes.append(text_box) 
    
    print(f"There are {len(non_trash_boxes)} non trash boxes")

    for box in non_trash_boxes:
        ((x_min, y_min), (x_max, y_max)) = box 
        cropped_text_img = cropped_part[y_min:y_max, x_min:x_max]
        
        text, score = recognize_text(cropped_text_img)
        
        match = re.search(r"(.+)(['`.,/])([sS5])", text)
        
        if match:
            # Group 1 is the name part before the separator
            name = match.group(1).strip()
            
            # Clean up the name (remove any trailing symbols OCR might have added)
            name = re.sub(r'[^a-zA-Z0-9 ]', '', name).lower()
            
            cv2.imwrite(f"names/{name}.png", cropped_text_img)
            return name

    return None

    


# === Main function ===
def get_trader_name(image, threshold=0.9, resized_threshold=0.85, max_workers=8):
    """Detect trader name using config first, then fallback to threaded search."""
    name_templates = fnmatch.filter(os.listdir(os.path.join(os.path.dirname(__file__), "names")), '*.png')

    height, width = image.shape[:2]
    square_size = height // 2
    cropped = image[0:square_size, square_size:width]

    # === Step 1: Try preferred template from config ===
    preferred_name = get_config("trader_name")
    preferred_accuracy = get_config("trader_accuracy")
    preferred_method = get_config("trader_method")  # "original" or "resized"
    preferred_scale = get_config("trader_scale")    # only valid if resized

    if preferred_name and preferred_accuracy and preferred_method:
        template_path = os.path.join(os.path.join(os.path.dirname(__file__), "names"), preferred_name)

        if preferred_method == "original":
            match, val = match_template_with_best(cropped, template_path, threshold)
        else:  # resized method
            match, val  = match_template_with_best_resized(cropped, template_path,preferred_scale, resized_threshold)
            

        if match is not None and np.isclose(val, float(preferred_accuracy), rtol=1e-3, atol=1e-4):
            Print(f"✅ Config template {preferred_name} matched with {preferred_method}, acc={val:.3f}")
            return preferred_name.split('.')[0]
        else:
            Print(f"⚠ Config template {preferred_name} failed (val={val:.3f}) preffered accuracy is {preferred_accuracy}, checking others...")

    # === Step 1: Parallel search (original size) ===
    all_matches = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_original, name, cropped, threshold): name for name in name_templates}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_matches.append(result) # result is (res_name, val, scale)

    if all_matches:
        # Sort by val (accuracy) in descending order and pick the top one
        best_name, best_val, best_scale = max(all_matches, key=lambda x: x[1])
        
        Print(f"✔ Best Template found: {best_name} with accuracy {best_val:.3f}")
        # Save to config and return...
        update_config("trader_name", best_name)
        update_config("trader_accuracy", best_val)
        update_config("trader_method", "original")
        update_config("trader_scale", best_scale)
        return best_name.split('.')[0]

    # === Step 2: Parallel search (resized templates) ===
    all_resized_matches = []
    SUPER_THRESHOLD = 0.98  # If we hit this, we stop looking

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_resized, name, cropped, resized_threshold): name for name in name_templates}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                res_name, val, scale = result
                all_resized_matches.append(result)
                
                # OPTIONAL: Short-circuit if we find a near-perfect match
                if val >= SUPER_THRESHOLD:
                    break 

    if all_resized_matches:
        # Pick the winner based on the highest accuracy (val)
        best_name, best_val, best_scale = max(all_resized_matches, key=lambda x: x[1])
        
        Print(f"🏆 Winner: {best_name} (Acc: {best_val:.3f}, Scale: {best_scale})")
        
        # Save to config
        update_config("trader_name", best_name)
        update_config("trader_accuracy", best_val)
        update_config("trader_method", "resized")
        update_config("trader_scale", best_scale)

        return best_name.split('.')[0]


    name = get_ocr_name(image)
    return name if name else "unknown"

# if __name__ == "__main__":
#     get_x_logo()