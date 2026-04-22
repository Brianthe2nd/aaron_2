import cv2
import os
import pandas as pd

logo_template = cv2.imread("templates/x_logo.png")
pair_search_template = cv2.imread("templates/search_button.png")
data = []
image_paths = os.listdir("images")
for image_path in image_paths:
    image = cv2.imread(os.path.join("images",image_path))
    """ calculate the template matching between the image and the two templates and save it a csv"""
    logo_result = cv2.matchTemplate(image,logo_template,cv2.TM_CCOEFF_NORMED)
    min_val, logo_max_val, min_loc, max_loc = cv2.minMaxLoc(logo_result)

    pair_search_result = cv2.matchTemplate(image,pair_search_template,cv2.TM_CCOEFF_NORMED)
    min_val, pair_max_val, min_loc, max_loc = cv2.minMaxLoc(pair_search_result)

    obj = {"image":image_path,
           "logo_match":logo_max_val,
           "pair_search_match":pair_max_val}
    data.append(obj)

# print(data)
df = pd.DataFrame(data)
df.to_csv("results.csv",index=False)
print("done")