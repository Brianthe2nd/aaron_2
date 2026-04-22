# from screen import capture_screen
import cv2
frame = 5000
import os

import matplotlib.pyplot as plt

def display_image(image_array, title="Image Preview", size=(10, 8), grid=False):
    """
    Displays an image array using Matplotlib.
    Handles BGR to RGB conversion automatically.
    """
    
    if image_array is None or len(image_array) == 0:
        print("Error: Image array is empty or None.")
        return
    # rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    # Create the figure
    plt.figure(figsize=size)
    
    # Check if the image is Grayscale or Color
    if len(image_array.shape) == 3:
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        # This flips the first and third channels
        rgb_image = image_array[:, :, ::-1]
        plt.imshow(rgb_image)
    else:
        # Display grayscale image
        plt.imshow(image_array, cmap='gray')
    
    plt.title(title)
    
    if not grid:
        plt.axis('off') # Hide the X and Y axis pixel coordinates
    
    plt.show()

# files = os.listdir()
# for file in files:
#     if file.endswith("mp4") or file.endswith("webm"):
#         with open("video_path.txt","w") as f:
#             f.write(file)
#         print(file)

        

#         image = capture_screen(None,frame_number= frame,exact= True)
#         if "Dakota" in file:
#             print("File",file)
#             print("saving the image")
#             cv2.imwrite("image.jpg",image)
#         # display_image(image,file)
#         h,w = image.shape[:2]
#         print(f"video: {file} , the size of the image is h, {h} w, {w}")

import csv
import re

def extract_debug_frames(txt_file_path, output_csv_path):
    """
    Reads a text file, extracts frame number, trader, and pair
    from lines containing '[DEBUG] Unknown signal count',
    and saves the results to a CSV file.
    """

    # Regex pattern based on your example line
    pattern = re.compile(
        r"-Frame\s+(?P<frame>\d+)-.*Unknown signal count for trader\s+(?P<trader>\w+),\s+pair\s+(?P<pair>\w+)"
    )

    extracted_rows = []

    with open(txt_file_path, "r", encoding="utf-8") as file:
        for line in file:
            if "Frame" in line and "[DEBUG] Unknown signal count" in line:
                match = pattern.search(line)
                if match:
                    extracted_rows.append([
                        match.group("frame"),
                        match.group("trader"),
                        match.group("pair")
                    ])

    # Write results to CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_number", "trader", "pair"])
        writer.writerows(extracted_rows)


# txt_f = "collected_files_20251230_090212_8acef9/logs.txt"
# csv_f = "d.csv"
# extract_debug_frames(txt_f,csv_f)
# import pandas as pd

# df = pd.read_csv("d.csv")
# for index, row in df.iterrows():
#     pair = row["pair"]
#     trade = row["trader"]
#     frame = row['frame_number']
#     frame_image = capture_screen(mon=None,frame_number=frame,exact=True)
#     print("PAIR: ",pair)
#     print("TRADER: ",trade)
#     print("FRAME NUMBER: ",frame)
#     print("\n")
#     display_image(frame_image)


if __name__ == "__main__":
    from main import crop_all_screen
    image = cv2.imread("fram.png")
    # cropped = crop_all_screen(image)
    # display_image(cropped)
    from logo import detect_best_logo_height
    a,b,c = detect_best_logo_height(image)
    print("best_height: ",a)
    print("best sim: ",b)
    print("location: ",c)