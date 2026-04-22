import io
import numpy as np
from PIL import Image
import cairosvg
import cv2
from resize import resize_proportional
from similar import similarity
from config import get_config
# from name import ocr

# from common_color import find_most_common_color
import os
# from paddle_inf import recognize_text
# def ocr(image):
#     return None

def svg_to_numpy_array(bg_color=(0,0,0), path_color=(255, 255, 255)):
    """
    Convert SVG to NumPy array with custom background and path colors.

    Parameters:
        bg_color: tuple(int, int, int) — Background color in RGB
        path_color: tuple(int, int, int) — Path (icon) color in RGB
    """
    # Convert RGB tuples to hex for SVG
    bg_hex = '#%02x%02x%02x' % bg_color
    path_hex = '#%02x%02x%02x' % path_color


    svg_data = f'''
    <svg xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 18 18"
        width="18"
        height="18">
        <path fill="{path_hex}"
            d="M3.5 8a4.5 4.5 0 1 1 9 0 4.5 4.5 0 0 1-9 0ZM8 2a6 6 0 1 0 3.65 10.76l3.58 3.58 1.06-1.06-3.57-3.57A6 6 0 0 0 8 2Z"/>
    </svg>
    '''


    # Convert SVG to PNG (in memory)
    png_bytes = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))

    # Load PNG into Pillow and convert to RGBA
    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    # Convert to NumPy array
    return np.array(image)

def prepare_for_match(img):
    # Convert to uint8 if necessary
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Handle images with alpha (4 channels)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # If you want grayscale matching (optional, faster and more tolerant)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def get_pair_img_search(point, image):
    x_1 = point[0][0]
    x_2 = point[1][0]
    
    y_1 = point[0][1]
    y_2 = point[1][1]
    
    sym_x_1 = x_2
    sym_x_2 = x_2 + 60
    sym_y_1 = y_1 - 10
    sym_y_2 = y_2 + 10
    
    img = image[sym_y_1:sym_y_2, sym_x_1:sym_x_2]

    height = img.shape[0]
    # rect_color = find_most_common_color(img)
    # print(f"rect_color: {rect_color}")
    # rect = np.full((height, 10, 3), rect_color, dtype=np.uint8)
    # out = np.hstack((rect, img))

    return img

def get_resize_height(logo_height):
    # 39 = 14
    # 49 = 18 
    # logo_height =
    return int((logo_height * 18) / 49)


def get_pair_search_img(image, logo_height,logo_loc,name):
    cv2.imwrite("search.png",image)
    print("logo heght: ",logo_height)
    print("logo loc: ",logo_loc)
    # from dump import display_image
    # display_image(image,"search function")
    # Load or generate template
    img_path = os.path.join(os.path.dirname(__file__), "templates", "search_button.png")
    if os.path.exists(img_path):
        search_button = cv2.imread(img_path)
    else:
        search_button = svg_to_numpy_array(bg_color=(42, 41, 46), path_color=(255, 255, 255))
        cv2.imwrite(img_path, cv2.cvtColor(search_button, cv2.COLOR_RGBA2BGRA))
        search_button = cv2.imread(img_path)

    h, w = image.shape[:2]
    top_left_image = image[0:h//3, 0:w//3]
    # cv2.imshow("",top_left_image)
    # cv2.waitKey(0)
    

    t_h, _ = search_button.shape[:2]
    orig_template = search_button.copy()

    main_points = []
    highest_sim = 0
    # best_h = get_resize_height(logo_height)
    # print(f"The name is: {name}")
    used_search = get_config("used_search",name = name)
    if used_search:
        best_h = get_config("best_search_logo_height",name= name)
        # print(f"Best height is: {best_h}")
    else:
        best_h = get_resize_height(logo_height)
    
    print("The best height search height in search.py is: ",best_h)
    # best_h = 8
    orig_template = resize_proportional(search_button,height = best_h)
    # cv2.imwrite("search_aaron.png",orig_template)
    highest_sim,main_points = similarity(top_left_image, orig_template,False)
    print("highest s: ",highest_sim)
    if not main_points :
        print("Search pair img in search.py: No match found with the best match.Looping through all sizes ...")
        for height in range(max(5, best_h - 4), best_h + 4):
            resized_template = resize_proportional(orig_template, height=height)
            s,points = similarity(top_left_image, resized_template,False)
            # print("s: ",s)
            # print("Height: ",height)
            # print("similarity: ",s)
            # print("len of points: ",len(points))
            # print("\n")
            if s > highest_sim and points:
                highest_sim = s
                main_points = points
                best_h = height
                

    print(f"Best height for search button is: {best_h}")

    if not main_points:
        print("No match found.")
        return None
    print("Number of search points is :",len(main_points))
    print(main_points)
    main_points_filtered = main_points.copy()
    top_left , bottom_right = logo_loc
    logo_bottom_y = bottom_right[1]
    if len(main_points) > 1:
        for point in main_points:
            x_1 = point[0][0]
            y_1 = point[0][1]
            x_2 = point[1][0]
            y_2 = point[1][1]
            
            if y_1 < logo_bottom_y:
                main_points_filtered.remove(point)
                print("removed the point")
            # cv2.rectangle(top_left_image,(x_1,y_1),(x_2,y_2),(0,255,0),4)
    
    # cv2.imshow("pair search matches",image)
    # cv2.waitKey()
    # from dump import display_image
    # display_image(top_left_image,"search image")
    
    point = main_points_filtered[0]
    return get_pair_img_search(point, image)

def get_sym_search_resize_width(logo_height):
    # 58 = 157 
    # logo_height =
    return int((logo_height * 157) / 58)

# def ocr():
#     return "sj"


if __name__ == "__main__":
    pair_search = svg_to_numpy_array()
    cv2.imwrite("templates/pair_search.png",pair_search)
    