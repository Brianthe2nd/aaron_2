from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
from pprint import pprint
from resize import resize_proportional
from similar import similarity

from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
from pprint import pprint
from resize import resize_proportional
from similar import similarity

def create_hash(size=256, circle_color=(255, 105, 180),  # pinkish red
                         bg_color=(46, 41, 42), line_width=25):
    """
    Creates a circular 'no entry' style icon similar to the uploaded image.

    Args:
        size (int): Width and height of the icon in pixels.
        circle_color (tuple): RGB color of the circle and slash.
        bg_color (tuple): RGB background color.
        line_width (int): Thickness of the circle border and slash.
    """
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    # Outer circle
    margin = line_width // 2
    draw.ellipse(
        (margin, margin, size - margin, size - margin),
        outline=circle_color,
        width=line_width,
        fill=(255,255,255)
    )

    # Diagonal slash
    offset = size * 0.2
    draw.line(
        (offset, offset, size - offset, size - offset),
        fill=circle_color,
        width=line_width
    )

    # img.show()
    return img

def create_blue_hash(size=256, circle_color=(255, 105, 180),  # pinkish red
                         bg_color=(46, 41, 42), line_width=25):
    """
    Creates a circular 'no entry' style icon similar to the uploaded image.

    Args:
        size (int): Width and height of the icon in pixels.
        circle_color (tuple): RGB color of the circle and slash.
        bg_color (tuple): RGB background color.
        line_width (int): Thickness of the circle border and slash.
    """
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    # (np.uint8(195), np.uint8(112), np.uint8(60))
    # blue_fill = (195, 112, 60)
    blue_fill = (60,112,195)
    # Outer circle
    margin = line_width // 2
    draw.ellipse(
        (margin, margin, size - margin, size - margin),
        outline=blue_fill,
        width=line_width,
        fill=blue_fill
    )

    # # Diagonal slash
    # offset = size * 0.2
    # draw.line(
    #     (offset, offset, size - offset, size - offset),
    #     fill=circle_color,
    #     width=line_width
    # )

    # img.show()
    return img


results=[]

if __name__ == "__main__":
    image = cv2.imread("C:/Users/Brayo/Desktop/topstep_revision/images/0.jpg")
    height, width = image.shape[:2]
    height = 24
    for l in range(20,50,5):
        for h in range(13 ,21):
            try:
                image = cv2.imread("C:/Users/Brayo/Desktop/topstep_revision/images/0.jpg")
                h2, w2 = image.shape[:2]
                image = image[0:h2//2 , 0:w2]
                label = create_hash(line_width=l)
                label_np = cv2.cvtColor(np.array(label.convert("RGB")), cv2.COLOR_RGB2BGR)
                label_np = resize_proportional(label_np,height = h) 
                cv2.imwrite("hash.png",label_np)
                s,_ = similarity(image,template=label_np,handle_blur=True)
                result = {"line_width":l,"height":h,"similarity":s}
                pprint(result)
                print("\n")
                results.append(result)
            except Exception as e:
                # pass
                print(f"Error: {e}")
                print(f"error at width: {l} , height: {h}")
    
    
    df = pd.DataFrame(results)
    df = df.sort_values(["similarity"])
    df.to_csv("hash.csv")
import numpy as np

import cv2
import numpy as np

def confirm_hash_colors_match(img_bgr,
                        border_rgb=(255, 105, 180),
                        fill_rgb=(255, 255, 255),
                        border_hue_tol=10):
    """
    Checks whether the hash template's border color and fill color
    are present in the detected hash image.

    Uses HSV ranges so antialiasing does not break detection.
    """

    # Convert inputs to HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    border_hsv = cv2.cvtColor(
        np.uint8([[border_rgb[::-1]]]),
        cv2.COLOR_BGR2HSV
    )[0][0]

    # ---------------------------------------
    # ✅ BORDER COLOR (pink / red hue range)
    # ---------------------------------------
    border_hue = int(border_hsv[0])
    lower_hue = max(0, border_hue - border_hue_tol)
    upper_hue = min(179, border_hue + border_hue_tol)

    # Mask for border color (hue + moderate saturation)
    border_mask = cv2.inRange(
        img_hsv,
        np.array([lower_hue, 80, 80]),
        np.array([upper_hue, 255, 255])
    )
    border_present = np.any(border_mask > 0)

    # ---------------------------------------
    # ✅ FILL COLOR (white inside circle)
    # ---------------------------------------
    # White = low saturation, high value
    fill_mask = cv2.inRange(
        img_hsv,
        np.array([0, 0, 200]),
        np.array([255, 255, 255])
    )
    fill_present = np.any(fill_mask > 0)

    # ---------------------------------------
    # ✅ RESULT
    # ---------------------------------------
    return bool(border_present and fill_present)

import cv2
import numpy as np

def confirm_blue_hash_color(img_bgr, 
                            fill_rgb=(195, 112, 60),
                            hue_tol=12):
    """
    Confirms that the detected image contains the expected blue-hash color
    using HSV hue tolerance matching. Since border and fill use the same
    color, only one color check is required.

    Returns True if the hue range is found anywhere in the image.
    """

    # Convert ROI to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Convert the fill color to HSV
    target_hsv = cv2.cvtColor(
        np.uint8([[fill_rgb[::-1]]]),  # RGB → BGR
        cv2.COLOR_BGR2HSV
    )[0][0]

    target_hue = int(target_hsv[0])

    # Hue range
    lower_hue = max(0, target_hue - hue_tol)
    upper_hue = min(179, target_hue + hue_tol)

    # Mask: hue must match, saturation & value must be non-zero
    mask = cv2.inRange(
        hsv,
        np.array([lower_hue, 40, 40]),
        np.array([upper_hue, 255, 255])
    )

    # ✅ If any pixel matches the hue → valid hash
    return np.any(mask > 0)


results=[]

if __name__ == "__main__":
    image = cv2.imread("C:/Users/Brayo/Desktop/topstep_revision/images/0.jpg")
    height, width = image.shape[:2]
    height = 24
    for l in range(20,50,5):
        for h in range(13 ,21):
            try:
                image = cv2.imread("C:/Users/Brayo/Desktop/topstep_revision/images/0.jpg")
                h2, w2 = image.shape[:2]
                image = image[0:h2//2 , 0:w2]
                label = create_hash(line_width=l)
                label_np = cv2.cvtColor(np.array(label.convert("RGB")), cv2.COLOR_RGB2BGR)
                label_np = resize_proportional(label_np,height = h) 
                cv2.imwrite("hash.png",label_np)
                s,_ = similarity(image,template=label_np,handle_blur=True)
                result = {"line_width":l,"height":h,"similarity":s}
                pprint(result)
                print("\n")
                results.append(result)
            except Exception as e:
                # pass
                print(f"Error: {e}")
                print(f"error at width: {l} , height: {h}")
    
    
    df = pd.DataFrame(results)
    df = df.sort_values(["similarity"])
    df.to_csv("hash.csv")
    
