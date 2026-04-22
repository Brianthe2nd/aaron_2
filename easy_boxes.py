
import easyocr
import numpy as np
import cv2
from paddle_inf import recognize_text
import os

reader = None

def get_boxes(image):
    '''
    Parameters:
    image: file path or numpy-array or a byte stream object
    '''

    reader = get_reader()
    horizontal_list, free_list = reader.detect(image)
    
    return horizontal_list,free_list

def get_reader():
    global reader
    if reader is None:
        # Note: If you only need boxes, recognizer=False is efficient
        reader = easyocr.Reader(lang_list=['en'], gpu=False,recognizer=False)
    return reader

def sym_trash_name(trader):
    if not os.path.exists(os.path.join("pair_templates",trader,"sym_trash")):
        os.mkdir(os.path.join("pair_templates",trader,"sym_trash"))
    
    sym_trash_paths = os.listdir(os.path.join("pair_templates",trader,"sym_trash"))

    for i in range(0,1000):
        if f"sym_trash_{i}.png" in sym_trash_paths:
            continue
        else:
            return os.path.join("pair_templates",trader,"sym_trash",f"sym_trash_{i}.png")

def in_sym_search(img, logo_height, trader):
    reader = get_reader()

    ideal_width  = (logo_height * 165) // 52     
    ideal_height = (logo_height * 25)  // 52

    horizontal_list, free_list = reader.detect(img)

    if not horizontal_list:
        return False

    rects = []
    for box in horizontal_list[0]:
        xmin, xmax, ymin, ymax = map(int, box)
        h = ymax - ymin
        w = xmax - xmin

        if h >= ideal_height * 0.8 and w >= ideal_width * 0.8:
            rects.append(((xmin, ymin), (xmax, ymax)))

    non_trash = []

    trash_dir = os.path.join("pair_templates", trader, "sym_trash")
    os.makedirs(trash_dir, exist_ok=True)
    sym_trash_paths = os.listdir(trash_dir)
    print(f"There are {len(rects)} detected trashes")
    img_h,img_w = img.shape[:2]

    for (xmin, ymin), (xmax, ymax) in rects:
        cropped = img[ymin:ymax, xmin:xmax]

        search_area = img[
            max(0, ymin - 5):min(ymax + 5,img_h),
            max(0, xmin - 5):min(xmax + 5,img_w)
        ]

        s_height, s_width = search_area.shape[:2]
        is_trash = False

        for sym_trash_path in sym_trash_paths:
            tpl = cv2.imread(os.path.join(trash_dir, sym_trash_path))
            if tpl is None:
                continue

            t_height, t_width = tpl.shape[:2]
            if t_height > s_height or t_width > s_width:
                continue

            res = cv2.matchTemplate(search_area, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > 0.95:
                is_trash = True
                break

        if not is_trash:
            non_trash.append(cropped)
    
    print(f"There are {len(non_trash)} non trash images")

    for idx, rect_img in enumerate(non_trash):

        text, score = recognize_text(rect_img)

        if text:
            text_l = text.lower()
            if "symbol" in text_l or "search" in text_l:
                return True
            else:
                cv2.imwrite(
                    sym_trash_name(trader),
                    rect_img
                )

    return False



if __name__ == "__main__":
    import time 
    import cv2
    start = time.time()
    img = cv2.imread("image_strip.png")
    # import easyocr
    # reader = easyocr.Reader(lang_list=['en'],gpu=False,recognizer=False)
    # img = cv2.imread('chinese_tra.jpg')

    f = in_sym_search("frame.jpg",52,"Dee")
    print(f)
    print(f"This took: {time.time() - start}")
    # h,f = get_boxes(image = image)
    # [[[np.int32(520), np.int32(544), np.int32(4), np.int32(10)]
    #   , [np.int32(33), np.int32(81), np.int32(5), np.int32(17)],
    #     [np.int32(102), np.int32(228), np.int32(6), np.int32(14)],
    #       [np.int32(232), np.int32(276), np.int32(6), np.int32(14)],
    #         [np.int32(554), np.int32(584), np.int32(6), np.int32(14)],
    #           [np.int32(632), np.int32(804), np.int32(6), np.int32(14)]]]

    # for box in h[0]:
    #     x1,x2,y1,y2 = box
    #     # print()
    #     cv2.rectangle(image , (x1,y1),(x2,y2),(0,255,0),1)
    # cv2.imshow("boxes",image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # print(h)
    # print(f)
    # print(f"This func took {time.time() - start}")