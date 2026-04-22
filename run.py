
from main import process_frame,create_or_append_number
from mt5_functions import init
from screen import capture_screen,capture_live_screen
import json
import os
import time
import csv
from std_out import Print
import random
from send_data import send_zipped_file,collect_and_zip_files
import sys
import shutil
import traceback
from datetime import datetime
import cv2




def init_trades_log():
    # CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
    CSV_FILE = os.path.join(os.path.dirname(__file__), "trades_2_log.csv")
    # # Ensure CSV file has a header if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pair', 'trader', 'event', 'timestamp' ,'yt_time','link','screens','frame','title'])




def create_data():
    trades_data = {
        "example": {
            "active": {
                random.choice(["GBPUSD", "USDJPY"]): {
                    "trade_type": random.choice(["buy", "sell"]),
                    "open_time": time.time(),
                    "sl": True,
                    "tp": True,
                    "status": "open"
                }
            },
            "unknown": {},
            "rejected": {},
            "config": {
                "ignore": True,
                "ignore_pairs": ["EURUSD"],
                "only_pairs": ["GBPUSD", "USDJPY"],
                "use_custom_risk": True,
                "custom_risk": 0.02
            }
        },
        "jd": {
            "active": {},
            "unknown": {},
            "rejected": {},
            "config": {
                "ignore": False,
                "ignore_pairs": [],
                "only_pairs": [],
                "use_custom_risk": True,
                "custom_risk": 0.2
            }
        },
        "doom": {
            "active": {},
            "unknown": {},
            "rejected": {},
            "config": {
                "ignore": False,
                "ignore_pairs": [],
                "only_pairs": ["US100"],
                "use_custom_risk": False,
                "custom_risk": 0
            }
        },
        "aaron": {
            "active": {},
            "unknown": {},
            "rejected": {},
            "config": {
                "ignore": False,
                "ignore_pairs": ["NG"],
                "only_pairs": [],
                "use_custom_risk": False,
                "custom_risk": 0
            }
        }
    }
    trades_data_file = os.path.join(os.path.dirname(__file__),"trades_data.json")
    with open(trades_data_file, "w") as f:
        json.dump(trades_data, f, indent=4)

# Example usage
# create_data()


def main():
    video_path_file = os.path.join(os.path.dirname(__file__),"video_path.txt")
    try:
        video_files = [f for f in os.listdir(os.path.dirname(__file__)) if f.endswith((".mp4", ".mkv", ".webm"))]
        if not video_files:
            raise FileNotFoundError("No video found in the csv_study folder")
        
        if not os.path.exists(video_path_file):
            with open(video_path_file, "w") as f:
                f.write(os.path.join(os.path.dirname(__file__),video_files[0]))

        # init(path)
        stream_is_live = True
        stream_link = ""
        stream_name = ""
        time_s = 0
        stream_mode = "low"
        active_trades = {}
        path = "C:/Program Files/FBS MetaTrader 5/terminal64.exe"
        # path = "C:/Program Files/MetaTrader 5/terminal64.exe"
        check_double_screen =  True
        crop_screen = True
        name = False
        check_x_scale = True
        """for when the x scale is changing in size"""
        trader_does_not_have_logo = False
        check_paper_acc = True
        check_limit_orders = False
        create_new_trade_data_file = True
        info_file = os.path.join(os.path.dirname(__file__), "info.json")
        with open(info_file,"r") as file:
            info = json.load(file)
            
        name = info.get("video_name").lower()
        if "jay" in name:
            name = "Jay"
            crop_screen = True
            trader_does_not_have_logo = False
        elif "dee" in name:
            name = "Dee"
            crop_screen = True
            trader_does_not_have_logo = False
        elif "aaron" in name:
            name = "Aaron"
            crop_screen = True
            trader_does_not_have_logo = False
        elif "dakota" in name:
            name = "Dakota"
            crop_screen = True
            trader_does_not_have_logo = False
        elif "anne" in name:
            name = "Marie"
            crop_screen = True
            trader_does_not_have_logo = False
        else:
            name = "all"
            crop_screen = False
            trader_does_not_have_logo = True

        
        

        # Start Wi-Fi monitor in background
        # start_wifi_thread("itel P55 5G")
        print("The name in run.py is: ",name)

        # Single JSON file for all trades
        if create_new_trade_data_file:
            create_data()
        trades_file = os.path.join(os.path.dirname(__file__),"trades_data.json")

        if not os.path.exists(trades_file):
            with open(trades_file, "w") as f:
                json.dump({}, f)

        count = 2
        csv_line = 2
        import csv

        def get_csv_line(file_path, line_number):
            """
            Returns the data from a specific line in a CSV file.
            Line numbers start at 1.
            """
            if line_number < 1:
                raise ValueError("Line number must be 1 or greater")

            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for current_line, row in enumerate(reader, start=1):
                    if current_line == line_number:
                        return row

            return None  # Line number not found


        while True:
            # internet_available.wait()  # Block here if no internet

            if stream_is_live:
                start=time.time()
                if not os.path.exists("frame_number.txt"):
                    with open("frame_number.txt", "w") as f:
                        f.write("0")
                with open("frame_number.txt","r") as file:
                    count = int(file.read())
                # print("Count: ",count)
                # count = 12450
                # pair,trader,trade,time_,g,f,g,frame,l = get_csv_line("trades_2_log.csv",csv_line) 
                # frame = 497910
                # csv_line = csv_line+1
                # frame = 82500
                # print("frame: ",frame)
                image = capture_screen(None , int(count) )
                if len(image) == 0:
                    print("The image does not exist")
                else:
                    print("Image is not corr")
                print("captured the screen")
                count = count + 1
                screen_num_file = os.path.join(os.path.dirname(__file__), "screen_num.txt")
                create_or_append_number(screen_num_file,0)
                if len(image) == 0:
                    break

                try:
                    with open(trades_file, "r") as f:
                        trades_data = json.load(f)
                except json.JSONDecodeError:
                    trades_data = {}

                trades_data = process_frame(
                    image,
                    time_s=time_s,
                    video_link=stream_name,
                    trades_data=trades_data,  # pass single dict
                    stream_mode=stream_mode,
                    check_double_screen=check_double_screen,
                    crop_screen=crop_screen,
                    name=name,
                    check_x_scale=check_x_scale,
                    check_paper_acc=check_paper_acc,
                    check_limit_orders=check_limit_orders,
                    trader_does_not_have_logo=trader_does_not_have_logo
                )
                print("finished processing the frame")
                # cv2.imwrite("frame.png",image)
                # cv2.namedWindow("l")
                # cv2.imshow("l",image)
                # cv2.resizeWindow("l",1080,720)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                # Save updated data back to single JSON file
                with open(trades_file, "w") as f:
                    json.dump(trades_data, f, indent=2)
                Print("Processing this image took: ",time.time()-start)
                Print("\n")
                # # pair,trader,trade,
                # print("Pair: ",pair)
                # print("Trader: ",trader)
                # print("Trade type: ",trade)
                # print("\n")

                # from dump import display_image
                # display_image(image)

        # 1. Read video path
        video_path_file = os.path.join(os.path.dirname(__file__),"video_path.txt")
        if not os.path.exists(video_path_file):
                raise FileNotFoundError(f"Video path file '{video_path_file}' not found.")
        with open(video_path_file, "r") as f:
            video_path = f.read().strip()
        # delete_file(video_path)
        folder, zip_file = collect_and_zip_files()
        Print("saved the zip file in: ",zip_file)
        send_zipped_file(zip_file)
        video_path_file = os.path.join(os.path.dirname(__file__),"video_path.txt")
        # from dump import display_image
        # display_image(image,"run image")


    except Exception as e:
        Print(e)
        Print(traceback.format_exc())
        # cv2.namedWindow("l")
        # cv2.imshow("l",image)
        # cv2.resizeWindow("l",1280,720)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        if not os.path.exists(video_path_file):
            raise FileNotFoundError(f"Video path file '{video_path_file}' not found.")
        with open(video_path_file, "r") as f:
            video_path = f.read().strip()
        # delete_file(video_path)




def delete_file(file_path: str) -> bool:
    """
    Deletes a file if it exists.

    Args:
        file_path (str): Path to the file to be deleted.

    Returns:
        bool: True if the file was deleted, False if it didn't exist.
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            Print(f"[DELETED] {file_path}")
            return True
        except Exception as e:
            Print(f"[ERROR] Could not delete {file_path}: {e}")
            return False
    else:
        Print(f"[SKIPPED] {file_path} does not exist.")
        return False


def archive_trade_logs(video_path_file):
    """
    Creates a folder and moves specific trade log files into it,
    then deletes them from the current folder (via move).

    Args:
        destination_root (str): Base directory where the archive folder will be created.
    """
    # List of files to archive
    files_to_move = [
        os.path.join(os.path.dirname(__file__),"trade_log.csv"),
        os.path.join(os.path.dirname(__file__),"trades_2_log.csv"),
        os.path.join(os.path.dirname(__file__),"logs.txt"),
        os.path.join(os.path.dirname(__file__),"mt5_errors.txt"),
        os.path.join(os.path.dirname(__file__),"active_trades.json"),
        os.path.join(os.path.dirname(__file__),"errors.txt")
    ]

    # Create destination folder with timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_folder = video_path_file.split("/")[-1].split(".")[0]
    dest_folder = os.path.join(os.path.dirname(__file__),"archive",dest_folder)
    os.makedirs(dest_folder, exist_ok=True)

    # Move files if they exist
    for filename in files_to_move:
        if os.path.exists(filename):
            shutil.move(filename, os.path.join(dest_folder, filename))
            Print(f"[MOVED] {filename} -> {dest_folder}")
        else:
            Print(f"[SKIPPED] {filename} not found.")

    Print(f"Archive completed: {dest_folder}")



if __name__ == "__main__":
    # video = sys.argv[1]  # full path passed in from subprocess
    main()
