from youtube import check_stream_is_live, get_video_title,download_stream
from run import main
import pandas as pd
import json
import shutil
import os


if __name__ == "__main__":
    df = pd.read_csv('aaron_streams_2026.csv')
    for index, row in df.iterrows():
        url = row['url']
        title = row['title']
        date = row['date']
        print(f"Checking stream: {title} ({url} on {date})")
        # print(f"Checking stream: {url}")
        print("Stream is LIVE! Downloading...")
        video_path = download_stream(url, title)
        print(f"Downloaded stream to: {video_path}")
        with open("video_path.txt", "w") as f:
            f.write(video_path)
        info = {"video_name": title, "video_path": video_path}
        json.dump(info, open("info.json", "w"), indent=4)
        main()





        # end 
        os.remove(video_path)
        os.remove("trades_data.json")
        os.remove("frame_number.txt")
        print(f"Removed video file: {video_path} from date: {date}")
        print("-" * 50)
        