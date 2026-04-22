import cv2
import numpy as np
import os
from std_out import Print,play_error_sound,log_exception
from logo import check_logo
from hash import create_hash,confirm_hash_colors_match,create_blue_hash,confirm_blue_hash_color
from similar import similarity
from resize import resize_proportional
from logo import detect_best_logo_height

def get_top_left(image):
    scales = np.linspace(0.2, 2, 50)
    logo_path = os.path.join(os.path.dirname(__file__), "templates", "logo.png")
    logo_exists, total_matches,matches = check_logo(image,scales,True,logo_path)
    Print("logo_exists:",logo_exists)
    if logo_exists:
        match_ = matches[0]
        top_left = match_[0]
    else:
        top_left = None
    
    return top_left,logo_exists,total_matches,matches

def get_top_right(or_image):
    scales = np.linspace(0.5, 2, 100)
    height, width = or_image.shape[:2]
    x_add = (3*(width // 5))
    y_add = height // 2
    image = or_image[0:y_add, x_add:width]
    top_right_path = os.path.join(os.path.dirname(__file__), "templates", "top_right.png")
    logo_exists, total_matches,matches = check_logo(image,scales,True,top_right_path,custom_image=True)
    if logo_exists:
        match_ = matches[0]
        top_right = match_[0]
        width = match_[1][1]
        x,y = top_right
        # cv2.rectangle(image, (x,y), (x+  20,y +10), (0, 255, 0), 2)
        # cv2.imshow("top_right",image)
        # cv2.waitKey(0)        
        top_right = (x+(width*2)+x_add,y)
        

    else:
        top_right = None
    
    return top_right

from skimage.metrics import structural_similarity as ssim

# ----------- CORE ANALYSIS FUNCTION -----------

def analyze_image_difference(original, modified):
    
    # cv2.imshow("original ",original)
    # cv2.imshow("modified ",modified)
    # cv2.imwrite("original.png",original)
    # cv2.imwrite("modified.png",modified)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    """Compute LAB differences: brightness, contrast and warmth between imgA and imgB and returns them the values when added to imgB should make imgB more similar to imgA."""
    
    lab1 = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(modified, cv2.COLOR_BGR2LAB)

    L1, A1, B1 = cv2.split(lab1)
    L2, A2, B2 = cv2.split(lab2)

    # Brightness diff
    delta_L = np.mean(L2) - np.mean(L1)

    # Contrast diff
    delta_contrast = np.std(L2) - np.std(L1)

    # Color/warmth diff
    delta_A = np.mean(A2) - np.mean(A1)
    delta_B = np.mean(B2) - np.mean(B1)

    # Structural similarity
    ssim_val = ssim(original, modified, channel_axis=2)
    # ssim_val = 0.5

    return delta_L, delta_contrast, delta_A, delta_B, ssim_val


# ----------- APPLY CORRECTIONS TO IMAGE A -----------

def apply_lab_correction(image, delta_L, delta_contrast, delta_A, delta_B):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    L = L.astype(np.float32)
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    # Brightness shift
    L += delta_L

    # Contrast adjustment (scaled to avoid overcorrection)
    mean_L = np.mean(L)
    contrast_scale = 1 + (delta_contrast / 50.0)  # divisor 50 makes it realistic
    L = (L - mean_L) * contrast_scale + mean_L

    # Warmth shifts
    A += delta_A
    B += delta_B

    # Clip
    L = np.clip(L, 0, 255).astype(np.uint8)
    A = np.clip(A, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    corrected = cv2.merge((L, A, B))
    corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    return corrected


# ----------- MAIN PIPELINE: MAKE A LOOK LIKE B -----------

def match_image(imgA, imgB):
    """Automatically adjust imgA so it matches imgB."""
    
    # Make sure both images are the same size
    imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

    # 1. Get differences
    dL, dContrast, dA, dB, ssim_before = analyze_image_difference(imgA, imgB)

    # 2. Apply correction to A
    imgA_corrected = apply_lab_correction(imgA, dL, dContrast, dA, dB)

    # 3. Measure similarity after correction
    _, _, _, _, ssim_after = analyze_image_difference(imgA_corrected, imgB)

    print("\n===== MATCHING SUMMARY =====")
    print(f"Brightness ΔL: {dL:.3f}")
    print(f"Contrast Δ:    {dContrast:.3f}")
    print(f"Warmth ΔA:     {dA:.3f}")
    print(f"Warmth ΔB:     {dB:.3f}")
    print(f"SSIM before:   {ssim_before:.4f}")
    print(f"SSIM after:    {ssim_after:.4f}")
    
    return imgA_corrected




def feature_match(image, method="SIFT", template_path=None, draw_matches=False, match_threshold=0.75):
    """
    Perform feature-based template matching using ORB or SIFT.
    It returns the image with the correct perspective boundary drawn.
    """
    # 1. Load Template (Using a placeholder path, replace with your actual logic)
    if template_path is None:
        # Use your original path logic if no path is passed
        template_path = os.path.join(os.path.dirname(__file__), "templates", "x_logo.png")
    
    template = cv2.imread(template_path)
    if template is None:
        print(f"Error: Could not load template from {template_path}")
        return {"found": False, "location": None, "matches_img": None}
        
    # 2. Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

    # 3. Choose detector and matcher
    if method.upper() == "SIFT":
        detector = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2)
    else: # ORB
        detector = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 4. Detect keypoints + descriptors
    kp1, des1 = detector.detectAndCompute(tpl_gray, None)
    kp2, des2 = detector.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {"found": False, "location": None, "matches_img": None}

    # 5. KNN Matching
    matches = bf.knnMatch(des1, des2, k=2)

    # 6. Lowe's Ratio Test (removes bad matches)
    good = []
    for m, n in matches:
        if m.distance < match_threshold * n.distance:
            good.append(m)
            
    if len(good) < 4:
        return {"found": False, "location": None, "matches_img": None}

    # 7. Compute Homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    if H is None:
        return {"found": False, "location": None, "matches_img": None}

    # 8. Warp template corners to image space
    h, w = tpl_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H) # Projected corners on the image

    # 9. Draw the correct perspective boundary
    # Use polylines to connect the 4 projected corners with lines (Yellow color: 0, 255, 255)
    img_with_box = cv2.polylines(image.copy(), [np.int32(projected)], True, (0, 255, 255), 3, cv2.LINE_AA)

    # 10. Optional: draw keypoint matches
    matches_img = None
    if draw_matches:
        matches_img = cv2.drawMatches(
            template, kp1, img_with_box, kp2, good, None,
            matchesMask=mask.ravel().tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        img_to_return = matches_img # Return the combined image if draw_matches is True
    else:
        img_to_return = img_with_box # Return the image with just the box

    # 11. Compute match center point (for debugging/alternative location return)
    # center_x = int(projected[:,0,0].mean())
    # center_y = int(projected[:,0,1].mean())
    # center = (center_x, center_y)

    # 12. Display and Return
    # NOTE: Moved image display and destruction outside the function for cleaner reuse
    # if draw_matches:
    #     cv2.imshow("Matches with Bounding Box", img_to_return)
    # else:
    #     cv2.imshow("Detected Logo", img_to_return)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return {
        "found": True,
        # Return the 4 corner points of the perspective-transformed boundary
        "location": np.int32(projected).reshape(-1, 2).tolist(),
        "matches_img": img_to_return
    }



def pre_process(image):
    h,w = image.shape[:2]
    cropped = image[:h//3 , :w//3]
    t_matching_result = detect_best_logo_height(cropped)
    if t_matching_result[0] != None:
        top_left,bottom_right = t_matching_result[2]
    else:
        result = feature_match(cropped)
        if result["found"]:
            top_left,bottom_right = result["location"]
    
    cropped_x_template = cropped[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    x_logo_path = os.path.join(os.path.dirname(__file__), "templates", "x_logo.png")
    x_template = cv2.imread(x_logo_path)
    x_template = cv2.resize(x_template, (cropped_x_template.shape[1], cropped_x_template.shape[0]))
    dL, dContrast, dA, dB, ssim_before = analyze_image_difference(cropped_x_template,x_template)
    corrected_image =  apply_lab_correction(image, dL, dContrast, dA, dB)
    corrected_template = apply_lab_correction(cropped_x_template, dL, dContrast, dA, dB)
    _, _, _, _, ssim_after = analyze_image_difference(corrected_template, x_template)


    print("\n===== MATCHING SUMMARY =====")
    print(f"Brightness ΔL: {dL:.3f}")
    print(f"Contrast Δ:    {dContrast:.3f}")
    print(f"Warmth ΔA:     {dA:.3f}")
    print(f"Warmth ΔB:     {dB:.3f}")
    print(f"SSIM before:   {ssim_before:.4f}")
    print(f"SSIM after:    {ssim_after:.4f}")
    return corrected_image
    

def get_bottom_right(or_image):
    scales = np.linspace(0.5, 2, 100)
    height, width = or_image.shape[:2]
    y_add = height // 2
    x_add = (3*(width // 5))
    image = or_image[ y_add: height, x_add:width]
    bottom_right_path = os.path.join(os.path.dirname(__file__), "templates", "bottom_right.png")

    logo_exists, total_matches,matches = check_logo(image,scales,True,bottom_right_path,custom_image=True)
    if logo_exists:
        match_ = matches[0]
        bottom_right = match_[0]
        width = match_[1][1]
        x,y = bottom_right
        bottom_right = (x+(width*2)+x_add,y+y_add)
        
        # cv2.rectangle(image, (x,y), (x+ + 20,y +10), (0, 255, 0), 2)
        # cv2.imshow("bottom_right",image)
        # cv2.waitKey(0)
    else:
        bottom_right = None
    
    return bottom_right

def get_bottom_left(image):
    bottom_left_template_path = os.path.join(os.path.dirname(__file__), "templates", "bl.png")
    # height, width = image.shape[:2]
    # image = image[height // 2 : height, 0:width//3]
    # cv2.imshow("bottom_left",image)
    # cv2.waitKey(0)
    def match_template_or_none(image, template_path):
        if not os.path.exists(template_path):
            # Print(f"Template not found: {template_path}")
            return None
        template = cv2.imread(template_path)
        if template is None:
            # Print(f"Failed to read template: {template_path}")
            return None
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        threshold = 0.7  # adjust as needed
        return max_loc if max_val > threshold else None
    
    loc = match_template_or_none(image, bottom_left_template_path)
    return loc

def get_resize_height(logo_height):
    # 39 = 12
    # 49 = 16 
    # logo_height =
    return int((logo_height * 16) / 49)


def crop_right(image,logo_height):
    h,w = image.shape[:2]
    template_path = os.path.join(os.path.dirname(__file__), "templates", "hash.png")
    blue_template_path = os.path.join(os.path.dirname(__file__), "templates", "blue_hash.png")
    resize_height = get_resize_height(logo_height)
    if not os.path.exists(template_path):
        label = create_hash(line_width=35)
        label_np = cv2.cvtColor(np.array(label.convert("RGB")), cv2.COLOR_RGB2BGR)
        template = resize_proportional(label_np,height=resize_height)
        cv2.imwrite(template_path,template)
    else:
        template = cv2.imread(template_path)
    if not os.path.exists(blue_template_path):
        label = create_blue_hash(line_width=35)
        label_np = cv2.cvtColor(np.array(label.convert("RGB")), cv2.COLOR_RGB2BGR)
        blue_template = resize_proportional(label_np,height=resize_height)
        cv2.imwrite(blue_template_path,blue_template)
    else:
        blue_template = cv2.imread(blue_template_path)

    s,points = similarity(image[0:h//3 , 0:w],template,0.8)
    s_blue,blue_points = similarity(image[0:h//3 , 0:w],blue_template,0.8)
    confirmed_points = []
    for p in points:
        x1 = p[0][0]
        y1 = p[0][1]
        x2 = p[1][0]
        y2 = p[1][1]
        hash_image = image[y1:y2,x1:x2]
        if confirm_hash_colors_match(hash_image):
            confirmed_points.append(p)
            # cv2.rectangle(image,(p[0][0],p[0][1]),(p[1][0],p[1][1]),(0,255,0),2)
    for p in blue_points:
        x1 = p[0][0]
        y1 = p[0][1]
        x2 = p[1][0]
        y2 = p[1][1]
        hash_image = image[y1:y2,x1:x2]
        if confirm_blue_hash_color(hash_image):
            confirmed_points.append(p)
            # cv2.rectangle(image,(p[0][0],p[0][1]),(p[1][0],p[1][1]),(0,255,0),2)

    print(confirmed_points)
    # img_path = "crop right image"
    # cv2.namedWindow(img_path, cv2.WINDOW_NORMAL)
    # cv2.imshow(img_path, image)
    # cv2.resizeWindow(img_path, 1280, 720)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    least_x = 99999999
    if len(confirmed_points) != 0:
        for point in confirmed_points :
            x1 = point[0][0]
            if x1 < least_x:
                least_x = x1
        # print(least_x)
        cropped_image = image[0:h,0:least_x - 10]
        return cropped_image
    else:
        return image
    

def crop_chart_region(image):
    img_height, img_width = image.shape[:2]
    image = crop_right(image)
    
    top_left,logo_exists,total_matches,matches = get_top_left(image)
    top_right = None
    bottom_left = get_bottom_left(image)
    bottom_right = None
    default_top = 0
    default_left = 0
    default_bottom = img_height
     
    if top_left is None:
        if bottom_left is not None:
            x1 = bottom_left[0]
            y1 = default_top
        else:
            x1 = default_left
            y1 = default_top
    
    else:
        x1, y1 = top_left
        
    if top_right is None:
        if bottom_right is not None:
            x2 = bottom_right[0]
        else:
            x2 = img_width
    else:
        x2 = top_right[0]
        
    if bottom_left is None:
        if bottom_right is not None:
            y2 = bottom_right[1]
        else:
            y2 = img_height
    else:
        y2 = bottom_left[1]
            

    # Ensure values are within image boundaries
    # x1 = max(0, min(x1, img_width))
    # x2 = max(0, min(x2, img_width))
    # y1 = max(0, min(y1, img_height))
    # y2 = max(0, min(y2, img_height))

    # if y2 <= y1 or x2 <= x1:
    #     cropped = image.copy()
    # else:
    cropped = image[y1:y2, x1:x2]


    return cropped,logo_exists,total_matches,matches




def reduce_close_points_exact(points, threshold=3):
    reduced = []
    used = [False] * len(points)

    for i, (x1, y1) in enumerate(points):
        if used[i]:
            continue
        reduced.append((x1, y1))
        used[i] = True

        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            if abs(x1 - x2) < threshold or abs(y1 - y2) < threshold:
                used[j] = True

    return reduced


def find_trade_buttons(trade_templates,chart_img, threshold=0.95):
    # Load the template
    available_trades={}
    for template in trade_templates:
      # if "sell_gray" in template or "buy_gray" in template:
      #   threshold = 0.8
      
      trade_template_path = os.path.join(os.path.dirname(__file__), "trade_templates", template)
      trade_template = cv2.imread(trade_template_path)
      if trade_template is None:
          raise FileNotFoundError(f"Template not found at {trade_template_path}")

      trade_h, trade_w = trade_template.shape[:2]

      # Perform template matching
      result = cv2.matchTemplate(chart_img, trade_template, cv2.TM_CCOEFF_NORMED)

      # Find all locations where match is above threshold
      y_coords, x_coords = np.where(result >= threshold)
      matches = list(zip(x_coords, y_coords))  # (x, y) format for OpenCV

      # Draw rectangles on the matches
    #   for (x, y) in matches:
    #       cs=cv2.rectangle(chart_img, (x, y), (x + trade_w, y + trade_h), (1,1,1), 1)
    #       cv2_imshow(cs)
      matches=reduce_close_points_exact(matches)

      # Print(f"The number of matches for {template.split('.')[0]} is {len(matches)}")
      # Print(matches)
      # if len(matches) != 0:
      #   available_trades.append(template.split(".")[0])
      available_trades[template.split(".")[0]] = len(matches)


      # previous_boxes = []

      # for trade_b in matches:
      #     previous_box_x = trade_b[0] - width  # shift left
      #     previous_boxes.append((previous_box_x, trade_b[1]))  # store as tuple for clarity

      # for (x, y) in previous_boxes:
      #     cs=cv2.rectangle(chart_img, (x, y), (x + trade_w, y + trade_h), (1,1,1), 1)
      #     cv2_imshow(cs)

      # Print("The coordinates of the previous boxes are:")
      # Print(previous_boxes)

    return available_trades


# for idx, (x, y) in enumerate(trade_buttons):
#     Print(f"Trade #{idx + 1} found at: ({x}, {y})")


# if __name__ == "__main__":
#     image = cv2.imread("screen.png")
#     from logo import detect_best_logo_height
#     logo,_ = detect_best_logo_height(image,cv2.imread("templates/x_logo.png"))
#     image = crop_right(image,logo)
    

if __name__ == "__main__":
    image = cv2.imread("l.png")
    r= feature_match(image,draw_matches=True)
    cv2.imshow("image",r["matches_img"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()