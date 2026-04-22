import cv2
# from trade_object import fetch_trades
from screen import capture_screen
import pandas  as pd
# from image_processing import crop_right
# from face import find_camera_box,detect_faces
# from logo import detect_best_logo_height
import numpy as np
from trade_object import fetch_trades
# from run import main
from run import process_frame
# from search import in_sym_search
from logo import detect_best_logo_height
from dump import display_image



df = pd.read_csv("collected_files_20260101_174932_629d2b/trades_2_log.csv")
for index, row in df.iterrows():
    pair = row["pair"]
    trade = row["trade"]
    frame = row['frame']
    frame_image = capture_screen(mon=None,frame_number=frame,exact=True)
    # logo_height , _ ,_ = detect_best_logo_height(frame_image )
    # print("in symbol search: ",in_sym_search(frame_image,logo_height))
    
    # display_image(frame_image)
    # logo_height = 52
    # cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
    # cv2.imshow("frame",frame_image)
    # cv2.resizeWindow("frame",1280,720)
    # cv2.waitKey()

    # cv2.imwrite("frame.jpg",frame_image)
    # trades_data = process_frame(
    #     frame_image,
    #     time_s=0,
    #     video_link="stream_name",
    #     trades_data={},  # pass single dict
    #     stream_mode="stream_mode",
    #     check_double_screen=False,
    #     crop_screen=False,
    #     name="Dakota",
    #     check_x_scale=False,
    #     check_paper_acc=False,
    #     check_limit_orders=False)
    # trades = fetch_trades(frame_image,logo_height)
    print("Pair: ",pair)
    print("Trade: ",trade)
    # print("Trades: ",trades)
    print("\n")
    # cv2.namedWindow("Camera Box Inference", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
    # cv2.imshow("frame",frame_image)
    # # cv2.resizeWindow("frame",1280,720)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    display_image(frame_image)

# 74550
# frame = row["frame"]
# frame = 59550
# frame = 39900
# # frame_number_file = "frame_number.txt"
# # with open(frame_number_file, "w") as f:
# #     f.write(str(frame))
# # print("Written the frame")
# # main()
# frame = capture_screen(mon=None,frame_number=frame,exact=True)
# cv2.imshow("frame",frame)
# cv2.waitKey()
# cv2.imwrite("test_frame.png",frame)

    
    # print(frame)
    # image = capture_screen("None",frame)

    # # aces = detect_faces(image)
    # # print(aces)
    # logo_template = cv2.imread("templates/x_logo.png")
    # # cv2.imshow("images",image)
    # logo_height,_ = detect_best_logo_height(image,logo_template=logo_template)
    # # print(logo_height)
    # image = crop_right(image,logo_height)
    # trades = fetch_trades(image,logo_height)
    # # image = cv2.rotate(image, cv2.ROTATE_180)
    # print(trades)
    # img_path = f"images/{frame}.jpg"
    # if frame == 73800:
    #     cv2.imwrite("frame.png",image)
    
    # cv2.namedWindow(img_path, cv2.WINDOW_NORMAL)
    # cv2.imshow(img_path, image)
    # cv2.resizeWindow(img_path, 1280, 720)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# (np.int32(102), np.int32(489), np.int32(760), np.int32(1013))]

# import time
# start = time.time()
# img = cv2.imread("screen.png")
# box = find_camera_box(img)
# end = time.time()
# IMG_PATH = "screen.png"
# print("Time taken:", end-start)
# print("Detection box:", box)
# x1,y1,x2,y2 = box[0]
# x_border = min(x1,x2)
# y_border = max(y1,y2)
# # cv2.rectangle(img, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 2)
# cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# cv2.namedWindow(IMG_PATH, cv2.WINDOW_NORMAL)
# cv2.imshow(IMG_PATH, img)
# cv2.resizeWindow(IMG_PATH, 1280, 720)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # {'x1': 110, 'y1': 702, 'x2': 545, 'y2': 1080, 'method': 'bbox'}

# image = cv2.imread("sell_in_profit.png")
# from trade_object import verify_trade_object_colors
# match =verify_trade_object_colors(image,"sell_in_profit")
# print(match)