# from screeninfo import get_monitors
# for m in get_monitors():
#     Print(str(m))

import mss
import cv2
import numpy as np
# from main import Print
from vidgear.gears import CamGear
import os
import imageio

def capture_screen(mon, frame_number, exact=None):
    """
    Rewritten to use imageio instead of OpenCV.
    """
    # 1. Setup paths
    base_dir = os.path.dirname(__file__)
    frame_number_file = os.path.join(base_dir, "frame_number.txt")
    video_path_file = os.path.join(base_dir, "video_path.txt")

    if not os.path.exists(video_path_file):
        raise FileNotFoundError(f"Video path file '{video_path_file}' not found.")
    
    with open(video_path_file, "r") as f:
        video_path = f.read().strip()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")

    # 2. Open video reader
    # 'ffmpeg' is the standard backend for mp4/avi etc.
    try:
        reader = imageio.get_reader(video_path, 'ffmpeg')
    except Exception as e:
        raise RuntimeError(f"Could not open video with imageio: {e}")

    # Get metadata
    meta = reader.get_meta_data()
    fps = meta.get('fps', 0)
    total_frames = meta.get('nframes', float('inf'))

    # 3. Handle frame number logic
    if not exact:
        if not os.path.exists(frame_number_file):
            with open(frame_number_file, "w") as f:
                f.write("0")
            current_idx = 0
        else:
            with open(frame_number_file, "r") as f:
                try:
                    current_idx = int(f.read().strip())
                except ValueError:
                    raise ValueError("Frame number file does not contain a valid integer.")
        
        # Calculate new frame (5 seconds ahead)
        frames_per_5s = int(fps * 5)
        frame_to_get = current_idx + frames_per_5s
    else:
        frame_to_get = frame_number

    # 4. Extract frame
    print(f"PROCESSING FRAME: {frame_to_get}")
    print("Total frames: ",total_frames)
    
    try:
        # Check bounds
        if frame_to_get >= total_frames:
            reader.close()
            return []

        # imageio allows direct indexing
        frame = reader.get_data(frame_to_get)
        
        # imageio returns RGB by default, while OpenCV uses BGR. 
        # If your downstream code expects BGR, uncomment the next line:
        # frame = frame[:, :, ::-1]
        
    except IndexError:
        # If frame is out of bounds
        if not exact:
            with open(frame_number_file, "w") as f:
                f.write("0")
        reader.close()
        return []
    finally:
        reader.close()

    # 5. Save updated frame number if not exact
    if not exact:
        with open(frame_number_file, "w") as f:
            f.write(str(frame_to_get))

    frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
    # from dump import display_image
    # display_image(frame,"cap screen") 
    return frame

# for i in range(0,100):
#     capture_screen(None,i*150)


if __name__ == "__main__":
    frame_number  = 148400
    frame = capture_screen(mon = None, frame_number= frame_number,exact= True)
    cv2.imwrite("screen_4.png",frame)
    from dump import display_image
    display_image(frame)
    # dee = cv2.imread("l.png")
    # cv2.imwrite("dee.png",dee)
    # dakota = cv2.imread("screen.png")
    # cv2.imwrite("dakota.png",dakota)
    

# def capture_screen(monitor_number=1):
#     with mss.mss() as sct:
        
#         # Get information of monitor 2
#         # monitor_number = 2
#         mon = sct.monitors[monitor_number]

#         # The screen part to capture
#         monitor = {
#             "top": mon["top"],
#             "left": mon["left"],
#             "width": mon["width"],
#             "height": mon["height"],
#             "mon": monitor_number,
#         }
#         # output = "sct-mon{mon}_{top}x{left}_{width}x{height}.png".format(**monitor)

#         # Grab the data
#         # sct_img = sct.grab(monitor)

#         img = np.array(sct.grab(monitor)) # BGR Image
#         # sct_img = sct.grab(monitor)
#         # cv2.imwrite("sct_mon_1.png", img)
#         Print("The image is {} pixels wide and {} pixels high".format(img.shape[1], img.shape[0]))
#         arr = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
#         if monitor_number == 2:
#             height , width = arr.shape[:2]
#             arr = arr[15+135:735+135,0:width]
        
#         # cv2.imwrite("ss_tabs_mon_2.png", arr)
        
#         # Save to the picture file
#         # output = "mss_ss.png"
#         # mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        
#         # return img
#         # Display the picture
#         # cv2.imshow("OpenCV", arr)
#         # cv2.waitKey(0)

#         return arr

def capture_live_screen(stream_link):
    options = {"STREAM_RESOLUTION": "720p"}
    stream = CamGear(source=stream_link, stream_mode = True, logging=True, **options).start()
    frame = stream.read()
    return frame 
    

# if __name__ == "__main__":
#     arr = capture_screen(2)  # Change the monitor number as needed
#     cv2.imwrite("screen.png",arr)
    # from main import scrape_screen
    # trades = scrape_screen(arr)
    # Print("The trades are ")
    # Print(trades)