import numpy as np
import cv2
import os
from batch_face import RetinaFace
from pair import get_easy_boxes
from paddle_inf import recognize_text



# ----------------------------
# FACE DETECTION
# ----------------------------

def detect_faces(img):
    detector = RetinaFace()

    # Ensure RGB input
    if isinstance(img, str):  
        img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = detector(
        img,
        threshold=0.95,
        resize=1,
        max_size=1080,
        return_dict=True
    )

    faces = []

    if isinstance(detections, list):
        for det in detections:
            box = det["box"]   # numpy array with 4 values
            faces.append(tuple(map(int, box)))  # convert to ints
    # print(f"There are {len(faces)} faces")
    return faces

# if __name__ == "__main__":
#     image = cv2.imread("frame.png")
#     import time
#     for i in range(5):
#         start = time.time()
#         detect_faces(image)
#         print(f"I: {i} , took: {time.time() - start}")
#         print("\n")
# ----------------------------
# CLASSIFY FACE LOCATION
# ----------------------------

import os
import cv2

def detect_faces_text(image):
    # 1. Setup Directories
    for folder in ["face_names", "name_trashes"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    image_h, image_w = image.shape[:2]
    x_end = (image_w * 366) // 1920
    y_start = (image_h * 117) // 1080
    y_end = (image_h * (1080 - 138)) // 1080
    
    faces_roi = image[y_start:y_end, 0:x_end]
    # from dump import display_image
    # display_image(faces_roi)
    f_h, f_w = faces_roi.shape[:2]

    # 2. Load Validation List
    names_list = [n.lower().split(".")[0] for n in os.listdir("names")]

    # 3. Load All Templates (Valid Names + Trash)
    all_templates = []
    
    # Load Valid Name Templates
    for n in os.listdir("face_names"):
        img = cv2.imread(f"face_names/{n}")
        if img is not None:
            all_templates.append({"img": img, "is_name": True})
            
    # Load Trash Templates
    for t in os.listdir("name_trashes"):
        img = cv2.imread(f"name_trashes/{t}")
        if img is not None:
            all_templates.append({"img": img, "is_name": False})

    horizontal_boxes = get_easy_boxes(faces_roi)
    face_count = 0

    for box in horizontal_boxes:
        # Early exit if we hit max faces
        if face_count >= 4:
            break

        x1, y1 = box[0]
        x2, y2 = box[1]
        
        # Search area for template matching
        search_area = faces_roi[max(0, y1-3):min(f_h, y2+3), max(0, x1-5):min(f_w, x2+5)]
        match_found = False

        # 4. Check against ALL known templates (Names and Trash)
        for template in all_templates:
            temp_img = template["img"]
            if temp_img.shape[0] > search_area.shape[0] or temp_img.shape[1] > search_area.shape[1]:
                continue

            res = cv2.matchTemplate(search_area, temp_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > 0.90:
                match_found = True
                if template["is_name"]:
                    face_count += 1
                break # Stop checking templates for this box
        
        # 5. If it's a new pattern, use OCR
        if not match_found:
            cropped = faces_roi[y1:y2, x1:x2]
            name_text, score = recognize_text(cropped)
            name_cleaned = str(name_text).lower().strip()

            if name_cleaned in names_list:
                # Save as a valid face name
                cv2.imwrite(f"face_names/{name_cleaned}.png", cropped)
                all_templates.append({"img": cropped, "is_name": True})
                face_count += 1
            else:
                # Save as trash to avoid OCRing this specific image pattern again
                # Using a timestamp or count to keep unique
                trash_id = len(os.listdir("name_trashes"))
                cv2.imwrite(f"name_trashes/trash_{trash_id}.png", cropped)
                all_templates.append({"img": cropped, "is_name": False})

    return face_count


        



def classify_face_region(img_w, img_h, fx1, fy1, fx2, fy2):
    cx = (fx1 + fx2) / 2
    cy = (fy1 + fy2) / 2

    left_th = img_w * 0.33
    right_th = img_w * 0.66
    top_th = img_h * 0.33
    bot_th = img_h * 0.66

    left = cx < left_th
    right = cx > right_th
    top = cy < top_th
    bottom = cy > bot_th

    if top and left:     return "top_left"
    if top and right:    return "top_right"
    if bottom and left:  return "bottom_left"
    if bottom and right: return "bottom_right"
    if top:              return "top"
    if bottom:           return "bottom"
    if left:             return "left"
    if right:            return "right"

    return "center"


# ----------------------------
# GET ROI BASED ON REGION
# ----------------------------
def get_roi_from_region(img, region):
    H, W = img.shape[:2]

    if region == "top_left":
        return img[0:int(H*0.5), 0:int(W*0.5)], 0, 0

    elif region == "top_right":
        return img[0:int(H*0.5), int(W*0.5):W], int(W*0.5), 0

    elif region == "bottom_left":
        return img[int(H*0.5):H, 0:int(W*0.5)], 0, int(H*0.5)

    elif region == "bottom_right":
        return img[int(H*0.5):H, int(W*0.5):W], int(W*0.5), int(H*0.5)

    elif region == "top":
        return img[0:int(H*0.5), :], 0, 0

    elif region == "bottom":
        return img[int(H*0.5):H, :], 0, int(H*0.5)

    elif region == "left":
        return img[:, 0:int(W*0.5)], 0, 0

    elif region == "right":
        return img[:, int(W*0.5):W], int(W*0.5), 0

    # fallback: whole image
    return img, 0, 0


def find_camera(img, region="bottom_left"):
    """
    Detects a rectangular camera box inside a livestream screenshot.

    Returns: (x1, y1, x2, y2) in full-image coordinates.
    """

    H, W = img.shape[:2]

    # ---- 1. Extract region of interest
    if region == "bottom_left":
        roi = img[int(H*0.50):H, 0:int(W*0.50)]
        offset_x, offset_y = 0, int(H*0.50)

    elif region == "bottom_right":
        roi = img[int(H*0.50):H, int(W*0.50):W]
        offset_x, offset_y = int(W*0.50), int(H*0.50)

    elif region == "top_left":
        roi = img[0:int(H*0.50), 0:int(W*0.50)]
        offset_x, offset_y = 0, 0

    elif region == "top_right":
        roi = img[0:int(H*0.50), int(W*0.50):W]
        offset_x, offset_y = int(W*0.50), 0

    else:
        raise ValueError("Invalid region name")

    # ---- 2. Gray → Canny edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 40, 120)

    # ---- 3. Hough Lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=30,
        maxLineGap=20
    )

    if lines is None:
        return None

    vertical = []
    horizontal = []

    # ---- 4. Separate vertical & horizontal
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 5:
            vertical.append((x1, y1, x2, y2))
        elif abs(y1 - y2) < 5:
            horizontal.append((x1, y1, x2, y2))

    if len(vertical) == 0 or len(horizontal) == 0:
        return None

    # ============================================================
    # ✅ 5. MERGE HORIZONTAL LINES WITH |ΔY| < 2
    # ============================================================
    horizontal.sort(key=lambda h: h[1])  # sort by Y value
    merged = []
    used = [False] * len(horizontal)

    for i, h in enumerate(horizontal):
        if used[i]:
            continue
        x1, y1, x2, y2 = h
        group = [(x1, y1, x2, y2)]
        used[i] = True

        for j in range(i + 1, len(horizontal)):
            X1, Y1, X2, Y2 = horizontal[j]
            if abs(Y1 - y1) < 2:  # same horizontal band
                used[j] = True
                group.append((X1, Y1, X2, Y2))

        # merge group: average Y, min-x to max-x
        ys = [g[1] for g in group]
        xs = [g[0] for g in group] + [g[2] for g in group]
        merged.append((min(xs), int(np.mean(ys)), max(xs), int(np.mean(ys))))

    horizontal = merged
    for line in merged:
        cv2.line(roi, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

    # cv2.imshow("roi", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ============================================================
    # ✅ 6. PICK INNER TOP / BOTTOM BASED ON REGION
    #    also enforce inner-line-length >= 60% max-length
    # ============================================================

    lengths = [abs(h[2] - h[0]) for h in horizontal]
    max_len = max(lengths)
    min_allowed = 0.60 * max_len

    def line_valid(h):
        return abs(h[2] - h[0]) >= min_allowed

    # top = smallest Y, bottom = largest Y
    top_candidates = sorted([h for h in horizontal if line_valid(h)], key=lambda h: h[1])
    bottom_candidates = sorted([h for h in horizontal if line_valid(h)], key=lambda h: h[1], reverse=True)

    if not top_candidates or not bottom_candidates:
        return None

    top_y = top_candidates[0][1]
    bottom_y = bottom_candidates[0][1]

    # ============================================================
    # ✅ 7. VERTICAL BORDER = image border based on region
    # ============================================================
    if region in ["top", "top_left", "top_right"]:
        top_y = 0
    if region in ["bottom", "bottom_left", "bottom_right"]:
        bottom_y = roi.shape[0]
    if region in ["top_left", "bottom_left" ,"left"]:
        left_x = 0
        right_x = max([v[0] for v in vertical])
    elif region in ["top_right", "bottom_right" ,"right"]:
        right_x = roi.shape[1]
        left_x = min([v[0] for v in vertical])
    else:
        # fallback
        left_x = min([v[0] for v in vertical])
        right_x = max([v[0] for v in vertical])

    # ============================================================
    # ✅ 8. Convert to full-image coordinates
    # ============================================================

    x1 = left_x + offset_x
    x2 = right_x + offset_x
    y1 = top_y + offset_y
    y2 = bottom_y + offset_y

    return (x1, y1, x2, y2)

def estimate_cam_from_face(face_box, W, H,region):
    """
    Estimate a webcam rectangle from a face bounding box and image size.
    Returns (x1, y1, x2, y2) in image coords (ints).
    """

    fx1, fy1, fx2, fy2 = map(int, face_box)
    face_w = fx2 - fx1
    face_h = fy2 - fy1

    # distances to edges
    dist_left  = fx1
    dist_right = W - 1 - fx2
    dist_top   = fy1
    dist_bot   = H - 1 - fy2

    # region = classify_face_region(W, H, fx1, fy1, fx2, fy2)
    
    # initialize
    x1 = 0
    x2 = W - 1
    y1 = 0
    y2 = H - 1

    if region == "bottom_left":
        # left border is image border
        x1 = 0
        # extend right by the distance from face left edge to left image border
        x2 = min(W - 1, fx2 + dist_left)
        # top a little above the face, bottom is image bottom
        y1 = max(0, fy1 - 50)
        y2 = H - 1

    elif region == "bottom_right":
        x2 = W - 1
        x1 = max(0, fx1 - dist_right)
        y1 = max(0, fy1 - 50)
        y2 = H - 1

    elif region == "top_left":
        x1 = 0
        x2 = min(W - 1, fx2 + dist_left)
        y1 = 0
        # bottom = 2 faces below the face + 10px
        y2 = min(H - 1, fy2 + 3 * face_h + 10)

    elif region == "top_right":
        x2 = W - 1
        x1 = max(0, fx1 - dist_right)
        y1 = 0
        y2 = min(H - 1, fy2 + 3 * face_h + 10)

    elif region == "left":
        # left border is image border, expand right by ~1 face width
        x1 = 0
        x2 = min(W - 1, fx2 + face_w)
        # vertical: extend one face up/down from detected face
        y1 = max(0, fy1 - face_h)
        y2 = min(H - 1, fy2 + face_h)

    elif region == "right":
        x2 = W - 1
        x1 = max(0, fx1 - face_w)
        y1 = max(0, fy1 - face_h)
        y2 = min(H - 1, fy2 + face_h)

    elif region == "top":
        # assume centered horizontally but top-aligned
        x1 = max(0, fx1 - face_w)
        x2 = min(W - 1, fx2 + face_w)
        y1 = 0
        y2 = min(H - 1, fy2 + 3 * face_h + 10)

    elif region == "bottom":
        x1 = max(0, fx1 - face_w)
        x2 = min(W - 1, fx2 + face_w)
        y1 = max(0, fy1 - 50)
        y2 = H - 1

    else:  # "center" or unknown
        # fallback: assume webcam ~ 2x face size around face center
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2
        half_w = int(face_w * 1.5)
        half_h = int(face_h * 1.5)
        x1 = max(0, cx - half_w)
        x2 = min(W - 1, cx + half_w)
        y1 = max(0, cy - half_h)
        y2 = min(H - 1, cy + half_h)

    # final sanity/clamp and ensure ordering
    x1 = int(max(0, min(x1, W - 1)))
    x2 = int(max(0, min(x2, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    y2 = int(max(0, min(y2, H - 1)))

    # ensure x1 < x2 and y1 < y2 (if not, nudge)
    if x2 <= x1:
        # expand to the right if possible
        x2 = min(W - 1, x1 + max(1, face_w))
        x1 = max(0, x2 - max(1, face_w))
    if y2 <= y1:
        y2 = min(H - 1, y1 + max(1, face_h))
        y1 = max(0, y2 - max(1, face_h))

    return (x1, y1, x2, y2)


def find_camera_box(img, visualize= False):
    H, W = img.shape[:2]

    faces = detect_faces(img)
    if not faces:
        print("No faces found.")
        return None

    results = []

    for (fx1, fy1, fx2, fy2) in faces:
        region = classify_face_region(W, H, fx1, fy1, fx2, fy2)
        print("Region detected:", region)
        
        face_box = (fx1, fy1, fx2, fy2)
        box = estimate_cam_from_face(face_box, W, H,region)
        if not box:
            continue

        results.append(box)

        # -------------- Visualization --------------
        if visualize:
            vis = img.copy()

            # Draw face
            cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

            # Draw inferred camera box
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.namedWindow("Camera Box Inference", cv2.WINDOW_NORMAL)
            cv2.imshow("Camera Box Inference", vis[:, :, ::-1])
            cv2.resizeWindow("Camera Box Inference", 1280, 720)
            cv2.waitKey(0)

    return results if results else None

if __name__ == "__main__":
    image = cv2.imread("frame.png")
    n_faces = detect_faces(image)
    t_faces = detect_faces_text(image)
    print("Normal faces: ",n_faces)
    print("Text faces: ",t_faces)