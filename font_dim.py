"""
This script will be used to get the range of the size of the futures pair when given the logo dimensions

1. Get the logo dimensions
2. Predict the height of the text from this logo dimensions
3. From the height of the text get the font size 
4. From the font size predict the min_width and the max width of the text
5. Return this range

image = 0.jpg
logo_height = 44

HGU25 . 15
height = 11
width = 77


image = 1000.jpg
logo_height = 47

MHGK25 . 30
height = 12
width = 93

44 = 11
47
"""
import cv2
from PIL import Image, ImageDraw, ImageFont

def measure_text_size(draw, text, font):
    """Return (width, height) of text using textbbox"""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def get_font_size(text, target_height, font_path="arial.ttf",
                                    max_font_size=200, min_font_size=5):
    """
    Adjusts font size so the text fits inside the given height.
    Returns: (font_size, text_width, text_height)
    """

    img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(img)

    font_size = max_font_size

    while font_size >= min_font_size:
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = measure_text_size(draw, text, font)

        if text_height <= target_height:
            return font_size, text_width, text_height

        font_size -= 1

    # If even smallest font is too tall, return it anyway
    return min_font_size, text_width, text_height


def get_text_height(logo_height):
    return round((logo_height * 11) / 44)

def get_text_range(logo_height):
    text_height = get_text_height(logo_height)
    addition = (logo_height * 15) // 57.2
    # 57.2 = 15
    logo_height
    min_text = "6II25"
    max_text = "MGGM25 . 500T"
    min_font_size,min_width,_ = get_font_size(min_text,text_height,"trebuchet-ms-2/trebuc.ttf")
    max_font_size,max_width,_ = get_font_size(max_text,text_height,"trebuchet-ms-2/trebuc.ttf")
    minus = round((max_font_size * 7) / 13)
    return (min_width-minus,max_width-minus+addition),text_height
