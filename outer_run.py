from youtube import check_stream_is_live, get_video_title,download_stream
from run import main
import pandas as pd
import json
import shutil
import os
import traceback
def row_processed(url):
    # loop through the collected folders and check if the date has been processed before
    files = os.listdir(os.path.dirname(__file__))
    for file in files:
        if file.startswith("collected_files") and os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
            inner_files = os.listdir(os.path.join(os.path.dirname(__file__), file))
            for inner_file in inner_files:
                if inner_file == "info.json":
                    info = json.load(open(os.path.join(os.path.dirname(__file__), file, inner_file)))
                    inner_url = info.get("url")
                    if inner_url == url:
                        print(f"Row with url: {url} has already been processed before. Skipping...")
                        return True

    return False


if __name__ == "__main__":
    df = pd.read_csv('aaron_streams_2026.csv')

    for index, row in df.iterrows():
        try:
            url = row['url']
            if row_processed(url):
                print(f"Row with url: {url} has already been processed before. Skipping...")
                continue
            title = row['title']
            date = row['date']
            line_number = row['line_number']

            print(f"Checking stream: {title} ({url} on {date})")
            # print(f"Checking stream: {url}")
            print("Stream is LIVE! Downloading...")
            video_path = download_stream(url, title)
            print(f"Downloaded stream to: {video_path}")
            with open("video_path.txt", "w") as f:
                f.write(video_path)
            info = {"video_name": title, "video_path": video_path ,"date": date , "url": url , "line_number": line_number}
            json.dump(info, open("info.json", "w"), indent=4)
            main()





            # end 
            os.remove(video_path)
            os.remove("trades_data.json")
            os.remove("frame_number.txt")
            os.remove("trades_2_log.csv")
            print(f"Removed video file: {video_path} from date: {date}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing row with url: {url} - {e}")
            print("-" * 50)
            traceback.print_exc()
            continue
            