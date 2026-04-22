import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def similarity(image,template,handle_blur = True,threshold = 0.6):
    if image is None or template is None:
        raise ValueError("One of the images could not be read — check file paths.")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if template.dtype != np.uint8:
        template = template.astype(np.uint8)
    def estimate_blur(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def match_blur(image, template):
        blur_img = estimate_blur(image)
        blur_tpl = estimate_blur(template)

        if blur_tpl > blur_img:
            blur_strength = np.sqrt(blur_tpl / blur_img)
            ksize = int(max(3, min(blur_strength * 3, 25)))  # limit kernel size
            template = cv2.GaussianBlur(template, (ksize | 1, ksize | 1), 0)

        elif blur_img > blur_tpl:
            blur_strength = np.sqrt(blur_img / blur_tpl)
            ksize = int(max(3, min(blur_strength * 3, 25)))
            image = cv2.GaussianBlur(image, (ksize | 1, ksize | 1), 0)
        return image, template
    if handle_blur == True:
        image, template = match_blur(image, template)


    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    locations = np.where(result >= threshold)
    points = list(zip(*locations[::-1]))  # (x, y) points
    _, highest_similarity, _, _ = cv2.minMaxLoc(result)
    min_distance  = 10
    template_h , template_w = template.shape[:2]
    filtered_points = []
    for pt in points:
        if all(np.linalg.norm(np.array(pt) - np.array(fp)) > min_distance for fp in filtered_points):
            filtered_points.append(pt)

    points_f =[]
    # print(f"The length of the filtered points is: {len(filtered_points)}")
    # print(filtered_points)
    if len(filtered_points) != 0 :
        for pt in filtered_points:
            x_min, y_min = pt
            x_max, y_max = x_min + template_w, y_min + template_h
            # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            point = ((x_min,y_min),(x_max, y_max))
            points_f.append(point)
    
    # cv2.namedWindow("D", cv2.WINDOW_NORMAL)
    # cv2.imshow("D", image)
    # cv2.resizeWindow("D", 1280, 720)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    return highest_similarity,points_f


# import pandas as pd

# df = pd.read_csv('font_similarity_results.csv')
# # sort the dataframe by similarity in descending order
# df = df.sort_values(by='similarity', ascending=False)
# df.to_csv("font_similarity_results_sorted.csv")
