"""We will modify this function so that the images are matched imediately and if we get a match the function further reduces the number
 of horizontal boxes so that the only boxes are those that touch or are within the image matched y coords
"""
import os
import cv2
import numpy as np

from pair import get_trash_name,get_easy_boxes,match_futures_symbol,sanitize_filename
from font_dim import get_text_range
from paddle_inf import recognize_text

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
    max_text_range = text_range[1]
    for box in horizontal_boxes:

        x1,y1 = box[0]
        x2,y2 = box[1]

        satisfies_range = False
        if (max_text_range > (x2 - x1)):
            satisfies_range = True
        if search :
            satisfies_range = True
        if satisfies_range:
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
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    x_1,y_1 = max_loc
                    temp_h , temp_w = trash_template.shape[:2]
                    x_2 = x_1 + temp_w   
                    y_2 = y_1 + temp_h   
                    # print("Match value:", max_val)
                    for i in range(0,200):
                        print("fuck ad adpai")
                    if max_val > 0.95:
                        # print("Trash found:", trash)
                        is_not_trash = False
                        break
                    else:
                        # print("This is not trash")
                        pass
                if is_not_trash:
                    non_trash_boxes.append(box)
            else:
                non_trash_boxes.append(box)

    print(f"There are {len(non_trash_boxes)} non trash boxes")
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
                print(f"Skipping {pair}: template larger than cropped image")
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
            unprocessed_boxes.append(box)

    # print(f"There are {len(unprocessed_boxes)} unprocessed boxes")
        
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
            cv2.imwrite(os.path.join("pair_templates",trader,trash_folder,get_trash_name(trader,trash_folder)),cropped_trash)
            """
                if a symbol is not matchable, crop it and save it in the trash folder 
            """
            # pass
    for symbol_data in symbols_data:
        for result in results:
            text = result[1]
            box = result[0]
            if text == symbol_data["raw"]:
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

                    if search and "/" in raw_text:
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


    return symbols_data if not search else symbols_data[0]

