import yt_dlp
from std_out import Print

def check_stream_is_live(url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': True,
        'forcejson': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # Check the 'is_live' flag or 'live_status'
            if info.get('is_live') or info.get('live_status') == 'is_live':
                return True
            else:
                return False
    except yt_dlp.utils.DownloadError as e:
        Print(f"[ERROR] yt-dlp failed: {e}")
        return False
    except Exception as e:
        Print(f"[ERROR] Something went wrong: {e}")
        return False

# cookies_path = 'cookies.txt'
# youtube_url = 'https://www.youtube.com/watch?v=YOUR_STREAM_ID'

# if check_stream_is_live(youtube_url, cookies_path):
#     Print("✅ Stream is LIVE!")
# else:
#     Print("❌ Stream is NOT live.")

import random
def download_stream(url, title):
    output_path = title + ".mp4"
    ydl_opts = {
        'quiet': True,
        'outtmpl': output_path,
        'format': 'bestvideo[ext=mp4]/bestvideo/best',  # Video-only formats
        'cookiefile': 'cookies/b603.txt',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            print(f"Stream downloaded successfully to {output_path}")
            return output_path
    except yt_dlp.utils.DownloadError as e:
        if "Requested format is not available" in str(e):
            print("[INFO] Best video format not available, trying alternative...")
            return download_video_only_fallback(url, title)
        else:
            print(f"[ERROR] yt-dlp failed: {e}")
    except Exception as e:
        print(f"[ERROR] Something went wrong: {e}")

def download_video_only_fallback(url, title):
    """Fallback function for video-only downloads"""
    output_path = title + ".mp4"
    
    # Try video-only formats in order of preference
    video_formats = [
        'bestvideo[ext=mp4]',     # Best MP4 video
        'bestvideo',               # Best video (any format)
        'best[ext=mp4]',          # Best MP4 (may include audio)
        'best',                    # Best overall (last resort)
    ]
    
    for fmt in video_formats:
        ydl_opts = {
            'quiet': False,
            'outtmpl': output_path,
            'format': fmt,
            'cookiefile': 'cookies/b603.txt',
        }
        
        try:
            print(f"[INFO] Trying video format: {fmt}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                print(f"Video downloaded successfully to {output_path}")
                return output_path
        except yt_dlp.utils.DownloadError:
            continue
    
    print("[ERROR] No video formats available")
    return None

def get_video_title(url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('title', 'Unknown Title')
    except yt_dlp.utils.DownloadError as e:
        Print(f"[ERROR] yt-dlp failed: {e}")
        return None
    except Exception as e:
        Print(f"[ERROR] Something went wrong: {e}")
        return None

# cookies_path = 'cookies.txt'
# video_url = 'https://www.youtube.com/watch?v=YOUR_VIDEO_ID'

# title = get_video_title(video_url, cookies_path)
# if title:
#     Print("🎬 Video title:", title)
# else:
#     Print("❌ Failed to retrieve title.")

# import subprocess
# import cv2
# import os
# import numpy as np

# def get_last_frame(video_id , chrome_profile_path="C:/Users/Brayo/AppData/Local/Google/Chrome/User Data/Default"):
#     try:
#         start = time.time()
#         result = subprocess.run(
#             ['yt-dlp', '-f', 'best[height=720]', '-g', video_id],
#             capture_output=True, text=True, check=True
#         )
#         video_url = result.stdout.strip().split('\n')[0]
#         Print(video_url)
#         Print("Getting video url took:", time.time() - start)

#         # Run ffmpeg to capture a single frame
#         subprocess.run(
#             ['ffmpeg', '-i', video_url, '-vframes', '1', 'last.jpg'],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#             check=True
#         )

#         # Read the image
#         frame = cv2.imread('last.jpg')
        
#         # Delete the image file
#         if os.path.exists('last.jpg'):
#             os.remove('last.jpg')
        
#         return frame

#     except subprocess.CalledProcessError as e:
#         Print("Subprocess error:", e)
#         return None
#     except Exception as ex:
#         Print("Error:", ex)
#         return None
# ffmpeg -i "$(yt-dlp -f "best[height=720]" -g --cookies-from-browser "chrome" [:C:/Users/Brayo/AppData/Local/Google/Chrome/User Data/Default]" 3kdVrCEinvo) -vframes 1 last.jpg

# yt-dlp -f "best[height=720]" -g 3kdVrCEinvo


import cv2
import time
if __name__ == "__main__":
    start  =time.time()
    title = get_video_title("https://www.youtube.com/watch?v=jFtIa_AEFiA")
    frame = check_stream_is_live("https://www.youtube.com/watch?v=jFtIa_AEFiA")
    Print("Running this frame took : ",time.time() - start)
    Print("Video title is : ",title)
    Print("Stream is live : ",frame)
    download_stream("https://www.youtube.com/watch?v=jFtIa_AEFiA", title)
    # cv2.imshow("frame",frame)
    # cv2.waitKey(0)
    