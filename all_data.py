from pprint import pprint
import cv2
import numpy as np
from trade_object import fetch_trades
from recreate_dot import process_points,find_green_dot
from logo import detect_best_logo_height
from color import is_main_color_white
from face import find_camera_box



def post_process_points(points):
    points.sort(key = lambda r: r[1],reverse = False)
    for i , point in enumerate(points):
        point_x = point[1]
        change = points[i+1][1] - point_x
        print(f"The change is: {change}")

def blackout_rectangles(img, rectangles):
    """
    Black out specified rectangles in an image and display the result.

    :param image_path: Path to the input image
    :param rectangles: List of rectangles in (x1, y1, x2, y2) format
    :return: Image with rectangles blacked out
    """
    # Load the image
    # img = cv2.imread(image_path)
    black = img.copy()
    if black is None:
        raise ValueError("Image not found or path is incorrect")

    # Black out each rectangle
    # for (x1, y1, x2, y2) in rectangles:
    #     cv2.rectangle(black, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    # Show the result
    # cv2.imshow('Blacked Out Image', black)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return black

# Example usage
# rects = [(50, 50, 150, 150), (200, 100, 300, 250)]
# blackout_rectangles('example.jpg', rects)
        

def get_data(image,logo_size,logo_loc,trader):
    height,width = image.shape[:2]
    # cv2.imshow("full_image",image)
    # cv2.waitKey()
    # results = find_green_dot(image,logo_loc)
    # print("Results")
    # print(results)
    top_rect = find_green_dot(image,logo_loc)
        
    screens = process_points(image,top_rect,logo_size,logo_loc,trader)
    print("Screens")
    print(screens)
    screens_data = []
    # screens = [screens[1:]]
    face_boxes = find_camera_box(image , False)
    if face_boxes:
        blacked = blackout_rectangles(image ,face_boxes)
    else:
        blacked = image.copy()

    
    
    for i,screen in enumerate(screens):
        if not screen["pred"]:
            print("\n")
            print(f"PROCESSING SCREEN: {i}")
            print("\n")
            pair = screen['pair']
            x_divider = screen['x_divider']
            screen_image = image[0:height ,x_divider:screens[i+1]['x_divider'] if i+1 < len(screens) else width]
            blacked_image = blacked[0:height ,x_divider:screens[i+1]['x_divider'] if i+1 < len(screens) else width]
            
            cv2.imwrite(f"screen_{i}.png",screen_image)
            # cv2.imshow(f"screen: {i}",screen_image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            whiteness = is_main_color_white(blacked_image)
            print(f"Screen {i}: {whiteness}%")
            if  whiteness < 20 if trader.lower() == "dee" else 35:
                # print("SCREEN WHITE : ",is_main_color_white(screen_image))
                trades = fetch_trades(screen_image,logo_size,logo_loc)
                print("The trades are :")
                print(trades)
                
                # cv2.line(image,(x_divider,0),(x_divider,height),(0,255,0),3)
                # print(trades)
                screen_data = {
                    'x_divider':x_divider,
                    'pair':pair,
                    'trades':trades
                }
                pprint(screen_data)

                screens_data.append(screen_data)
                
            else:
                print(f"skipping screen {i} because the image is allegedly mostly white")
    return screens_data


if __name__ == '__main__':
    image = cv2.imread('frame.png')
    logo_template = cv2.imread("templates/x_logo.png")
    logo_size,_,_,logo_loc = detect_best_logo_height(image,logo_template)
    screens = get_data(image,logo_size)
    pprint(screens)
    

# if __name__ == '__main__':
#     image = cv2.imread('images/0.jpg')
#     screens = get_data(image)
#     pprint(screens)