import cv2
import numpy as np

def feature_match(template, image, method="ORB", draw_matches=False, match_threshold=0.75):
    """
    Perform feature-based template matching using ORB or SIFT.
    
    Args:
        template (np.ndarray): Template (BGR or grayscale)
        image (np.ndarray): Full image (BGR or grayscale)
        method (str): "ORB" or "SIFT"
        draw_matches (bool): Return an image showing matched keypoints
        match_threshold (float): Lowe's ratio threshold (0.6–0.8 recommended)
    
    Returns:
        dict with:
            "found" : bool
            "location": (x, y) of the match center or None
            "matches_img": optional visualization image
    """

    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

    # Choose detector
    if method.upper() == "SIFT":
        detector = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2)
    else:
        detector = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Detect keypoints + descriptors
    kp1, des1 = detector.detectAndCompute(tpl_gray, None)
    kp2, des2 = detector.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        return {"found": False, "location": None, "matches_img": None}

    # KNN Matching
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's Ratio Test (removes bad matches)
    good = []
    for m, n in matches:
        if m.distance < match_threshold * n.distance:
            good.append(m)
    print("good: ",len(good))
    if len(good) < 4:
        # Not enough matches to compute homography
        return {"found": False, "location": None, "matches_img": None}

    # Collect matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Compute Homography (gives exact template location)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    # print("H: ",H)
    # cv2.imshow("Mask",mask)
    # cv2.waitKey(0)
    if H is None:
        return {"found": False, "location": None, "matches_img": None}

    # Warp template corners to image space
    h, w = tpl_gray.shape
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    projected = cv2.perspectiveTransform(corners, H)

    # Compute match center point
    center_x = int(projected[:,0,0].mean())
    center_y = int(projected[:,0,1].mean())
    center = (center_x, center_y)
    top_left , top_right , bottom_right , bottom_left = projected[0][0],projected[1][0],projected[2][0],projected[3][0]
    top_left_x = int(min(top_left[0],top_right[0],bottom_left[0],bottom_right[0]))
    top_left_y = int(min(top_left[1],top_right[1],bottom_left[1],bottom_right[1]))
    bottom_right_x = int(max(top_left[0],top_right[0],bottom_left[0],bottom_right[0]))
    bottom_right_y = int(max(top_left[1],top_right[1],bottom_left[1],bottom_right[1]))
    # Optional: draw matches + detection box
    matches_img = None
    print("projected:")
    print("Top left: X: ",top_left_x," Y: ",top_left_y)
    print("Bottom right: X: ",bottom_right_x," Y: ",bottom_right_y)
   
    # if draw_matches:
    #     # matches_img = cv2.drawMatches(
    #     #     template, kp1, image, kp2, good, None,
    #     #     matchesMask=mask.ravel().tolist(),
    #     #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    #     # )

    #     cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)


    return {
        "found": True,
        "location": ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)),
        "matches_img": image
    }

if __name__ == "__main__":
    template = cv2.imread("templates/x_logo.png")
    image = cv2.imread("frame.png")
    h,w = image.shape[:2]
    image = image[:h//4 , :w//6]
    result = feature_match(template, image, method="SIFT", draw_matches=True)

    if result["found"]:
        print("Match found at:", result["location"])
        # cv2.imshow("result", result["matches_img"])
        # cv2.waitKey(0)
    else:
        print("No match found.")
