import cv2 
import os 
import numpy as np
import time
import concurrent.futures
import traceback
from resize import resize_proportional
from std_out import Print
from color import orange_percentage
from config import get_config,update_config


def measure_x_logo(img):
    """
    Measures the X shape from a manually cropped logo image.
    Uses:
      - Method 1: contour bounding box from binary threshold
      - Method 4: tight nonzero pixel bounding box
    Confirms and adjusts so height ≈ 2 × width.
    Prints all individual measurements and returns final ones.
    """
    # Read image and convert to grayscale
    # img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Method 1: Threshold + Contour bounding box ---
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x1, y1, w1, h1 = cv2.boundingRect(c)

    print(f"Method 1  Width: {w1}px, Height: {h1}px, Ratio: {h1 / w1:.2f}")

    # --- Method 4: Tight bounding box on nonzero pixels ---
    rows = np.any(thresh, axis=1)
    cols = np.any(thresh, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min, y_max = y_indices[0], y_indices[-1]
    x_min, x_max = x_indices[0], x_indices[-1]
    w2 = x_max - x_min
    h2 = y_max - y_min

    print(f"Method 2  Width: {w2}px, Height: {h2}px, Ratio: {h2 / w2:.2f}")

    h_avg = int(round((h1 + h2) / 2))
    h_avg = h_avg + 1
    while True:
        if (h_avg % 2) == 0:
            h_avg = h_avg + 1
        else:
            break
    
    w_avg = int(h_avg / 2 )

    return {"width": w_avg, "height": h_avg}



def match_template_or_none(image, template_path, scale):
    if not os.path.exists(template_path):
        return None, []

    template = cv2.imread(template_path)
    if template is None:
        return None, []

    try:
        # resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        resized_template = resize_proportional(template , height = scale)
    except Exception as e:
        traceback.print_exc()
        Print(f"Resize error at scale {scale}: {e}")
        return None, []

    if resized_template.shape[0] > image.shape[0] or resized_template.shape[1] > image.shape[1]:
        return None, []

    result_match = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
    match_locations = np.where(result_match >= 0.85)
    points = list(zip(match_locations[1], match_locations[0]))  # (x, y)

    match_info = (scale, resized_template.shape[1], resized_template.shape[0])  # (scale, width, height)
    return match_info, points


def check_logo(image, scales=range(20,100), return_matches=False,
               logo_path=os.path.join(os.path.dirname(__file__), "templates", "logo.png"), custom_image=False, name = None):
    """
    Tries to find a logo in the given image at various scales.
    Uses stored scale from config if available, updates config with best scale.
    """
    if not custom_image:
        height, width = image.shape[:2]
        image = image[0:height // 2, 0:width // 2]

    start = time.time()
    all_matches = []

    # Step 1: Try preferred scale first (from config if exists)
    preferred_scale = get_config("logo_scale",name = name)
    if preferred_scale is not None:
        match_info, points = match_template_or_none(image, logo_path, preferred_scale)
        if points:
            for pt in points:
                all_matches.append((pt, match_info))
            Print(f"✅ Found logo match at preferred scale {preferred_scale}")
        else:
            Print(f"⚠ No logo match at preferred scale {preferred_scale}, checking other scales...")

    # Step 2: Try other scales if no matches found
    if not all_matches:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(match_template_or_none, image, logo_path, scale)
                for scale in scales if scale != preferred_scale
            ]
            for future in concurrent.futures.as_completed(futures):
                match_info, points = future.result()
                for pt in points:
                    all_matches.append((pt, match_info))

    # Step 3: Filter duplicates
    filtered_matches = []
    all_matches.sort(key=lambda x: x[1])  # sort by match_info
    for pt, match_info in all_matches:
        if all(abs(pt[0] - fp[0]) >= 8 and abs(pt[1] - fp[1]) >= 8 for fp, _ in filtered_matches):
            filtered_matches.append((pt, match_info))

    end = time.time()
    Print("Time taken:", round(end - start, 3), "seconds")

    # Step 4: Save best scale in config & return results
    # Print("The filtered matches are:",filtered_matches)
    if filtered_matches:
        best_scale = filtered_matches[0][1][0]  # take scale from first match_info
        update_config("logo_scale", best_scale)

        Print(f"🔢 Total distinct logos matched above 0.85: {len(filtered_matches)}")
        Print(f"💾 Saved best logo scale {best_scale} to config")
        if return_matches:
            return True, len(filtered_matches), filtered_matches
        else:
            return True, len(filtered_matches)
    else:
        Print(f"❌ No good logo match found for {logo_path}")
        if return_matches:
            return False, 0, filtered_matches
        else:
            return False, 0

# import cv2
# import numpy as np # Import numpy for array manipulation if not already present
def predict_logo_from_search(search_h, search_top_left, search_bottom_right):
    """
    Predicts logo height and logo bounding box from search button info.
    """
    print("search location")
    print("search top left")
    print(search_top_left)
    print("search bottom right")
    print(search_bottom_right)

    # --- Calibration values from your dataset ---
    avg_ratio = 2.600   # logo_height / search_height

    avg_dx = -72.7      # logo_x - search_x (reference scale)
    avg_dy = -67.3      # logo_y - search_y

    reference_search_h = 18  # used to normalize scale

    # ---------------------------------------------
    # SCALE FACTOR (UI elements scale proportionally)
    # ---------------------------------------------
    scale = search_h / reference_search_h

    # ---------------------------------------------
    # PREDICT LOGO HEIGHT (scaled)
    # ---------------------------------------------
    logo_h = search_h * avg_ratio


    # ---------------------------------------------
    # PREDICT LOGO POSITION
    # ---------------------------------------------
    sx, sy = search_top_left
    ex, ey = search_bottom_right

    # scaled offsets
    dx = avg_dx * scale
    dy = avg_dy * scale

    # top-left of logo
    logo_x1 = int(sx + dx)
    logo_y1 = int(sy + dy)

    # bottom-right uses height (logo is roughly square-ish)
    logo_x2 = int(logo_x1 + logo_h)
    logo_y2 = int(logo_y1 + logo_h)

    logo_data = {
        "logo_height": logo_h,
        "logo_top_left": (logo_x1, logo_y1),
        "logo_bottom_right": (logo_x2, logo_y2)
    }
    print("logo data")
    print(logo_data)
    return logo_data

# def detect_best_logo_height(full_image, logo_template=None, search_region=None):

#     def boxes_touch_or_overlap(a, b):
#         """Return True if bounding boxes overlap or even touch."""
#         (ax1, ay1), (ax2, ay2) = a
#         (bx1, by1), (bx2, by2) = b

#         # If one box is completely to the left/right OR above/below → NO touch
#         if ax2 < bx1 or bx2 < ax1:
#             return False
#         if ay2 < by1 or by2 < ay1:
#             return False
#         return True  # They overlap OR touch edges

#     def suppress_touching_boxes(sorted_matches):
#         """Remove touching/overlapping boxes, keeping only highest score."""
#         final = []
#         for m in sorted_matches:
#             touched = False
#             for kept in final:
#                 if boxes_touch_or_overlap(m["box"], kept["box"]):
#                     touched = True
#                     break
#             if not touched:
#                 final.append(m)
#         return final

#     # --- TEMPLATE LOAD ---
#     x_logo_template_path = os.path.join(os.path.dirname(__file__), "templates", "x_logo.png")
#     if logo_template is None:
#         logo_template = cv2.imread(x_logo_template_path)

#     # --- REGION SETUP ---
#     offset_x = 0
#     offset_y = 0

#     if search_region is None:
#         h_full, w_full = full_image.shape[:2]
#         search_region = full_image[:, : w_full // 2]

#     # --- BEST HEIGHT SEARCH ---
#     heights = range(20, 100)

#     best_sim = -1
#     best_h = None
#     best_loc = None
#     best_template_w = None
#     best_template_h = None

#     gray_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

#     for h in heights:
#         try:
#             resized = resize_proportional(logo_template, height=h)
#             gray_template = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

#             th, tw = gray_template.shape[:2]
#             res = cv2.matchTemplate(gray_region, gray_template, cv2.TM_CCOEFF_NORMED)
#             _, sim, _, loc = cv2.minMaxLoc(res)

#             if sim > best_sim:
#                 best_sim = sim
#                 best_h = h
#                 best_loc = loc
#                 best_template_w = tw
#                 best_template_h = th
#         except:
#             pass

#     # --- STOP IF NO GOOD MATCH ---
#     if best_sim <= 0.5 or best_loc is None:
#         print(f"❌ No confident match found. Best Sim: {best_sim:.4f}")
#         return None, None, None

#     # --- MULTI-MATCH DETECTION USING THE BEST HEIGHT ---
#     resized_best = resize_proportional(logo_template, height=best_h)
#     gray_template_best = cv2.cvtColor(resized_best, cv2.COLOR_BGR2GRAY)
#     th, tw = gray_template_best.shape[:2]

#     res = cv2.matchTemplate(gray_region, gray_template_best, cv2.TM_CCOEFF_NORMED)

#     # matches >= 90% of the best similarity
#     threshold = best_sim * 0.9
#     match_locations = np.where(res >= threshold)

#     all_matches = []
#     for (y, x) in zip(match_locations[0], match_locations[1]):
#         score = res[y, x]
#         box = (
#             (x + offset_x, y + offset_y),
#             (x + offset_x + tw, y + offset_y + th)
#         )
#         all_matches.append({"score": score, "box": box})

#     # Sort by score highest → lowest
#     all_matches.sort(key=lambda m: m["score"], reverse=True)

#     # --- FILTER OUT TOUCHING/OVERLAPPING DETECTIONS ---
#     distinct_matches = suppress_touching_boxes(all_matches)

#     print(f"Found {len(distinct_matches)} distinct non-touching matches.")

#     return best_h, best_sim, [m["box"] for m in distinct_matches]

def detect_best_logo_height(full_image, logo_template=None, search_region=None,name =None,trader_does_not_have_logo = False):
    print("detecting best logo size")
    # print("The full image is: ")
    # from dump import display_image
    # display_image(full_image)
    # cv2.imshow("full image in logo.py", full_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    def boxes_touch_or_overlap(a, b):
        (ax1, ay1), (ax2, ay2) = a
        (bx1, by1), (bx2, by2) = b
        if ax2 < bx1 or bx2 < ax1: return False
        if ay2 < by1 or by2 < ay1: return False
        return True

    def suppress_touching_boxes(sorted_matches):
        final = []
        for m in sorted_matches:
            if not any(boxes_touch_or_overlap(m["box"], kept["box"]) for kept in final):
                final.append(m)
        return final



    # --- REGION SETUP ---
    offset_x, offset_y = 0, 0
    if search_region is None:
        h_full, w_full = full_image.shape[:2]
        search_region = full_image[:h_full // 3,: w_full // 5]

    gray_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    best_sim, best_h, best_loc = -1, None, None
    best_template_w, best_template_h = None, None

    if not trader_does_not_have_logo:
        # --- TEMPLATE LOAD ---
        x_logo_template_path = os.path.join(os.path.dirname(__file__), "templates", "x_logo.png")
        if logo_template is None:
            logo_template = cv2.imread(x_logo_template_path)
        # --- BEST HEIGHT SEARCH WITH CACHE FOR X_LOGO ---
        print("The name in the detect best logo height is: ",name)
        cached_logo_height = get_config("best_logo_height",name= name)
        cached_logo_match = get_config("best_logo_match",name = name)
        heights = range(20, 100)


        # Try cached height first if available
        if cached_logo_height:
            try:
                resized = resize_proportional(logo_template, height=cached_logo_height)
                gray_template = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                th, tw = gray_template.shape[:2]
                res = cv2.matchTemplate(gray_region, gray_template, cv2.TM_CCOEFF_NORMED)
                _, sim, _, loc = cv2.minMaxLoc(res)
                if sim >= cached_logo_match - 0.025 and sim >= 0.8:
                    best_sim, best_h, best_loc = sim, cached_logo_height, loc
                    best_template_w, best_template_h = tw, th
                    print(f"✅ Using cached logo height: {cached_logo_height} with sim={sim:.3f}")
            except:
                pass

        # If cached height fails or not found, run the loop
        if best_sim < 0.5:
            print("Brute forcing the logo heights")
            for h in heights:
                try:
                    resized = resize_proportional(logo_template, height=h)
                    gray_template = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    th, tw = gray_template.shape[:2]
                    res = cv2.matchTemplate(gray_region, gray_template, cv2.TM_CCOEFF_NORMED)
                    _, sim, _, loc = cv2.minMaxLoc(res)
                    if sim > best_sim:
                        best_sim, best_h, best_loc = sim, h, loc
                        best_template_w, best_template_h = tw, th
                except:
                    pass

            # Save the best height for next time
            if best_h:
                update_config("best_logo_height", best_h , name = name )
                update_config("best_logo_match" , best_sim, name= name)
                update_config("used_search",False,name = name)

    # --- STOP IF NO GOOD MATCH AND TRY SEARCH BUTTON FALLBACK ---
    if best_sim < 0.7 or best_loc is None:
        search_template_path = os.path.join(os.path.dirname(__file__),"templates", "pair_search.png")
        if not os.path.exists(search_template_path):
            from search import svg_to_numpy_array
            search_template = svg_to_numpy_array(bg_color=(45,38,43))
            cv2.imwrite(search_template_path,search_template)
        else:
            search_template = cv2.imread(search_template_path)
        
        search_template = cv2.cvtColor(search_template,cv2.COLOR_BGR2GRAY)




        search_heights = range(10, 40)

        cached_search_logo_height = get_config("best_search_logo_height",name = name )  # cache for fallback
        cached_search_logo_match = get_config("best_search_logo_match",name = name)  # cache for fallback

        best_search_score, best_search_height = -1, None
        best_search_loc, best_search_template_h, best_search_template_w = None, None, None

        # Try cached search height first
        if cached_search_logo_height:
            try:

                temp_search = resize_proportional(search_template, height=cached_search_logo_height)
                res = cv2.matchTemplate(gray_region, temp_search, cv2.TM_CCOEFF_NORMED)
                _, score, _, max_loc = cv2.minMaxLoc(res)
                th, tw = temp_search.shape[:2]
                if score >= cached_logo_match - 0.06 :
                    best_search_score = score
                    best_search_height = cached_search_logo_height
                    best_search_loc = max_loc
                    best_search_template_h, best_search_template_w = th, tw
            
                    print(f"✅ Using cached search-button height: {cached_search_logo_height} with sim={score:.3f}")
                else:
                    print("Cached search logo failed")
            except:
                pass

        # If cached search fails, brute-force heights
        if best_search_score < 0.5:
            print("Brute forcing the search logo heights")
            for h_search in search_heights:
                try:
                    temp_search = resize_proportional(search_template.copy(), height=h_search)
                    res = cv2.matchTemplate(gray_region, temp_search, cv2.TM_CCOEFF_NORMED)
                    _, score, _, max_loc = cv2.minMaxLoc(res)
                    # print("Size: ",h_search)
                    # print("Score: ",score)
                    th, tw = temp_search.shape[:2]
                    if score > best_search_score:
                        best_search_score = score
                        best_search_height = h_search
                        best_search_loc = max_loc
                        best_search_template_h, best_search_template_w = th, tw
                except:
                    # pass
                    traceback.print_exc()
            if best_search_height is not None:
                print("The best search height was:", best_search_height)

                top_left = best_search_loc
                bottom_right = (
                    top_left[0] + best_search_template_w,
                    top_left[1] + best_search_template_h
                )

                # cv2.rectangle(
                #     search_region,
                #     top_left,
                #     bottom_right,
                #     (0, 255, 0),
                #     1
                # )

                # from dump import display_image
                # display_image(search_region, "logo search function")

                update_config("best_search_logo_height", best_search_height, name=name)
                update_config("best_search_logo_match", best_search_score, name=name)
                update_config("used_search",True,name = name)

        if best_search_score > 0.5 and best_search_loc is not None:
            # cv2.rectangle(full_image , best_search_loc ,(best_search_loc[0] + best_search_template_w, best_search_loc[1] + best_search_template_h),(0,255,0),1)
            logo_data = predict_logo_from_search(
                best_search_height,
                best_search_loc,
                (best_search_loc[0] + best_search_template_w, best_search_loc[1] + best_search_template_h)
            )

            best_h = logo_data.get("logo_height")
            best_loc = logo_data.get("logo_top_left")
            best_sim = best_search_score
            bottom_right = logo_data.get("logo_bottom_right")

            # Cache the search-logo height for future
            # update_config("best_search_logo_height", best_h, name = name )
            # update_config("best_search_logo_match", best_sim,name = name )
            
            return best_h, best_sim, [(best_loc, bottom_right)]
        else:
            print(f"❌ No confident match found. Best Sim: {best_sim:.4f}")
            return None, None, None

    # --- MULTI-MATCH DETECTION USING BEST HEIGHT ---
    print("The best sim is: ",best_sim)
    resized_best = resize_proportional(logo_template, height=best_h)
    gray_template_best = cv2.cvtColor(resized_best, cv2.COLOR_BGR2GRAY)
    th, tw = gray_template_best.shape[:2]
    res = cv2.matchTemplate(gray_region, gray_template_best, cv2.TM_CCOEFF_NORMED)
    threshold = best_sim * 0.9
    match_locations = np.where(res >= threshold)

    all_matches = [{"score": res[y, x],
                    "box": ((x + offset_x, y + offset_y), (x + offset_x + tw, y + offset_y + th))}
                   for y, x in zip(match_locations[0], match_locations[1])]

    all_matches.sort(key=lambda m: m["score"], reverse=True)
    distinct_matches = suppress_touching_boxes(all_matches)

    print(f"Found {len(distinct_matches)} distinct non-touching matches.")
    return best_h, best_sim, [m["box"] for m in distinct_matches]


# def detect_best_logo_height(full_image, logo_template= None, search_region=None):
#     """
#     Finds the best matching logo size and draws the best match on the image.
#     Returns the best height, similarity, and the image with the bounding box.
#     """
#     x_logo_template_path = os.path.join(os.path.dirname(__file__), "templates", "x_logo.png")
#     if logo_template is None:
#         logo_template = cv2.imread(x_logo_template_path)
#     # --- Initial Setup and Region Definition ---
    
#     # Store the offset of the search_region relative to full_image (top-left corner)
#     # Assume the region starts at (0, 0) of the full_image unless a custom region is passed
#     offset_x = 0
#     offset_y = 0
    
#     # Auto-crop region if none
#     if search_region is None:
#         h_full, w_full = full_image.shape[:2]
#         # Your original default region: the left half of the image
#         search_region = full_image[:, : w_full // 2] 
#         # offset_x remains 0, offset_y remains 0
#     else:
#         # If search_region is a slice or a copy, you need a way to determine 
#         # its coordinates in full_image if it's not the default.
#         # **NOTE:** If search_region is passed as a predefined region (e.g., full_image[y:y+h, x:x+w]), 
#         # you MUST adjust offset_x and offset_y accordingly for accurate drawing.
#         # For simplicity in this example, we'll rely on the default case, or 
#         # assume the user handles offset if a custom region is passed outside the default.
#         pass 

#     base_h = logo_template.shape[0]

#     # Height search range
#     heights = range(20 , 100)

#     best_sim = -1
#     best_h = None
#     # Variables to store the coordinates of the best match
#     best_loc = None
#     best_template_w = None
#     best_template_h = None


#     # Preprocess search region once
#     gray_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

#     # --- Loop for Best Match Height ---
    
#     for h in heights:
#         try:
#             # Resize logo correctly (preserve aspect ratio)
#             resized = resize_proportional(logo_template, height=h)

#             # Preprocess template
#             gray_template = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#             template_h, template_w = gray_template.shape[:2]

#             # Template match
#             res = cv2.matchTemplate(gray_region, gray_template,
#                                     cv2.TM_CCOEFF_NORMED)
#             _, sim, _, max_loc = cv2.minMaxLoc(res)
#             # max_loc is the top-left corner (x, y) of the best match in the search_region

#             # print(f"Height: {h}, Sim: {sim}")
#             if sim > best_sim:
#                 best_sim = sim
#                 best_h = h
#                 # Store the location and dimensions of the best match
#                 best_loc = max_loc
#                 best_template_w = template_w
#                 best_template_h = template_h
#         except:
#             pass
            
#     # --- Drawing the Bounding Box on the Image ---
    
#     if best_sim > 0.5 and best_loc is not None: # Use a threshold like 0.5 for confidence
#         # Calculate coordinates in the full_image
#         # Top-left corner
#         top_left_x = best_loc[0] + offset_x
#         top_left_y = best_loc[1] + offset_y
        
#         # Bottom-right corner
#         bottom_right_x = top_left_x + best_template_w
#         bottom_right_y = top_left_y + best_template_h
        
#         # Draw the rectangle on the full_image (Color: Green (0, 255, 0), Thickness: 3)
#         # Note: full_image is modified in place
#         # cv2.rectangle(full_image, 
#         #               (top_left_x, top_left_y), 
#         #               (bottom_right_x, bottom_right_y), 
#         #               (0, 255, 0), 
#         #               1)
        
#         # cv2.imshow("image",full_image)
#         # cv2.waitKey(0)
        
#         print(f"✅ Best match found and drawn! Height: {best_h}, Similarity: {best_sim:.4f}")
#         return best_h, best_sim, ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
#     else:
#         """Try getting the size of the search button"""
#         search_heights = range(8,40)
#         search_template = cv2.imread("templates/pair_search.png", cv2.IMREAD_GRAYSCALE)
#         scaled_searches = {h: resize_proportional(search_template.copy(), height=h)
#                     for h in search_heights}
#         best_search_score = -1
#         best_search_height = None

#         for h_search, temp_search in scaled_searches.items():
#             res = cv2.matchTemplate(search_region, temp_search, cv2.TM_CCOEFF_NORMED)
#             _, score, _, max_loc = cv2.minMaxLoc(res)
#             template_h, template_w = temp_search.shape[:2]

#             if score > best_search_score:
#                 best_search_score = score
#                 best_search_height = h_search
#                 best_search_loc = max_loc
#                 best_search_template_h = template_h
#                 best_search_template_w = template_w 

#         if best_search_score > 0.5 and best_search_loc is not None:
#             logo_data = predict_logo_from_search(best_search_height, best_search_loc, (best_search_loc[0] + best_search_template_w, best_search_loc[1] + best_search_template_h))
#             best_h = logo_data.get("logo_height")
#             best_loc = logo_data.get("logo_top_left")
#             best_sim = best_search_score
#             bottom_right = logo_data.get("logo_bottom_right")
#             return best_h, best_sim, (best_loc, bottom_right)

#         else:
#             print(f"❌ No confident match found. Best Sim: {best_sim:.4f}")
#             return None , None , None 

    # Return the image with the drawing as well as the original results

# def detect_best_logo_height(full_image, logo_template= None, search_region=None):
#     """
#     Finds the best matching logo size and draws the best match on the image.
#     Returns the best height, similarity, and the image with the bounding box.
#     """
#     x_logo_template_path = os.path.join(os.path.dirname(__file__), "templates", "x_logo.png")
#     if logo_template is None:
#         logo_template = cv2.imread(x_logo_template_path)
#     # --- Initial Setup and Region Definition ---
    
#     # Store the offset of the search_region relative to full_image (top-left corner)
#     # Assume the region starts at (0, 0) of the full_image unless a custom region is passed
#     offset_x = 0
#     offset_y = 0
    
#     # Auto-crop region if none
#     if search_region is None:
#         h_full, w_full = full_image.shape[:2]
#         # Your original default region: the left half of the image
#         search_region = full_image[:, : w_full // 2] 
#         # offset_x remains 0, offset_y remains 0
#     else:
#         # If search_region is a slice or a copy, you need a way to determine 
#         # its coordinates in full_image if it's not the default.
#         # **NOTE:** If search_region is passed as a predefined region (e.g., full_image[y:y+h, x:x+w]), 
#         # you MUST adjust offset_x and offset_y accordingly for accurate drawing.
#         # For simplicity in this example, we'll rely on the default case, or 
#         # assume the user handles offset if a custom region is passed outside the default.
#         pass 

#     base_h = logo_template.shape[0]

#     # Height search range
#     heights = range(20 , 100)

#     best_sim = -1
#     best_h = None
#     # Variables to store the coordinates of the best match
#     best_loc = None
#     best_template_w = None
#     best_template_h = None


#     # Preprocess search region once
#     gray_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

#     # --- Loop for Best Match Height ---
    
#     for h in heights:
#         try:
#             # Resize logo correctly (preserve aspect ratio)
#             resized = resize_proportional(logo_template, height=h)

#             # Preprocess template
#             gray_template = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#             template_h, template_w = gray_template.shape[:2]

#             # Template match
#             res = cv2.matchTemplate(gray_region, gray_template,
#                                     cv2.TM_CCOEFF_NORMED)
#             _, sim, _, max_loc = cv2.minMaxLoc(res)
#             # max_loc is the top-left corner (x, y) of the best match in the search_region

#             # print(f"Height: {h}, Sim: {sim}")
#             if sim > best_sim:
#                 best_sim = sim
#                 best_h = h
#                 # Store the location and dimensions of the best match
#                 best_loc = max_loc
#                 best_template_w = template_w
#                 best_template_h = template_h
#         except:
#             pass
            
#     # --- Drawing the Bounding Box on the Image ---
    
#     if best_sim > 0.5 and best_loc is not None: # Use a threshold like 0.5 for confidence
#         # Calculate coordinates in the full_image
#         # Top-left corner
#         top_left_x = best_loc[0] + offset_x
#         top_left_y = best_loc[1] + offset_y
        
#         # Bottom-right corner
#         bottom_right_x = top_left_x + best_template_w
#         bottom_right_y = top_left_y + best_template_h
        
#         # Draw the rectangle on the full_image (Color: Green (0, 255, 0), Thickness: 3)
#         # Note: full_image is modified in place
#         # cv2.rectangle(full_image, 
#         #               (top_left_x, top_left_y), 
#         #               (bottom_right_x, bottom_right_y), 
#         #               (0, 255, 0), 
#         #               1)
        
#         # cv2.imshow("image",full_image)
#         # cv2.waitKey(0)
        
#         print(f"✅ Best match found and drawn! Height: {best_h}, Similarity: {best_sim:.4f}")
#         return best_h, best_sim, ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
#     else:
#         print(f"❌ No confident match found. Best Sim: {best_sim:.4f}")
#         return None , None , None

#     # Return the image with the drawing as well as the original results
    

if __name__ == "__main__":
    image = cv2.imread("frame.jpg")
    template = cv2.imread("templates/x_logo.png")
    h,s,loc = detect_best_logo_height(image,template)  
    print(f"The best height is: {h} with a similarity of: {s}")
    # for image_path in ["aaron","dee","dakota","aaron_2"]:
    #     image = cv2.imread(f"{image_path}.png")
    #     template = cv2.imread("templates/x_logo.png")
    #     h,s,loc = detect_best_logo_height(image,template)   
    #     cv2.imshow(image_path, image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     print(f"The best height is: {h} with a similarity of: {s}")
    
    

# if __name__ == "__main__":
#     # image = cv2.imread("names/Screenshot 2025-08-07 154031.png")
#     # scales = np.linspace(0.3, 2, 50)
#     # logo_exists, total_matches,matches = check_logo(image,scales,True)
#     # Print("Logo exists:", logo_exists)
#     # Print("Total matches:", total_matches)
#     # Print("Matches:", matches)
#     # x,y = matches[0][0]
#     # _,width,height = matches[0][1] 
    
#     # x_type = (width//3) + width + x
#     # y_type = height//2 + y
#     # Print("The orange percentage is:",orange_percentage(image[y:y_type,x_type:x_type+width]))
#     # # cv2.imshow("Detected Logos", image[y:y_type,x_type:x_type+width])
#     # cv2.waitKey(0)
#     image = cv2.imread("farame.png")
#     logo_template = cv2.imread("templates/x_logo.png")
#     h,s = detect_best_logo_height(image,logo_template)
#     print(f"The height with the highest similarity of {s} is height: {h}")
#     # from search import get_pair_img_search
#     from search import get_pair_search_img
#     pair = get_pair_search_img(image=image,logo_height=h)
#     cv2.imshow("",pair)
#     cv2.waitKey()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     i = cv2.imread("screen.png")
#     l,_ = detect_best_logo_height(i,cv2.imread("templates/x_logo.png"))
#     print(l)
    