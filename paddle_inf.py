# from paddleocr import PaddleOCR
from paddleocr import TextDetection
from paddleocr import TextRecognition
import time

ocr = None   # global cache
text_detection_model = None
text_recognition_model = None

def get_ocr():
    return ocr

# def get_ocr():
#     global ocr
#     if ocr is None:
#         print("Initializing OCR...")
#         ocr = PaddleOCR(
#             use_doc_orientation_classify=False,
#             use_doc_unwarping=False,
#             use_textline_orientation=False,
#             # rec = False
#             text_recognition_model_name=None,  # 🔹 disables recognition
#             text_recognition_model_dir=None
#         )
#         ocr = 
#     return ocr

    # from paddleocr import TextDetection
    # model = TextDetection()
def get_text_detection_model():
    global text_detection_model 
    if text_detection_model is None:
        text_detection_model = TextDetection()
    
    return text_detection_model

def get_text_recognition_model():
    global text_recognition_model 
    if text_recognition_model is None:
        text_recognition_model = TextRecognition()
    
    return text_recognition_model

def recognize_text(image):
    text_recognition_model = get_text_recognition_model()
    output = text_recognition_model.predict(image ,batch_size = 1)
    print(output)
    text = output[0].get("rec_text") 
    score = output[0].get("rec_score")
    return text,score 

    

# Example usage:
if __name__ ==  "__main__":
    import cv2
    import time
    iss = ["pair_templates/Dee/search/MGC.png","pair_templates/Dee/search/MGC.png","pair_templates/Dee/search/MGC.png","lq.png"]
    # from paddleocr import TextDetection
    # model = get_text_recognition_model()
    for j,i in enumerate(iss):
        i = "image_strip.png"
        start = time.time()  
        # output = model.predict(i, batch_size=1)
        output = recognize_text(i)
        print(output)
        for res in output:
            res.print()
            res.save_to_img(save_path = f"image_{j}.png")
            res.save_to_json(save_path=f"res_{j}.json")
        print(f"This took : {time.time() - start} seconds")
        
    # for i in iss:
    #     image = cv2.imread(i) 

    #     run_ocr(image)
    #     print(f"This took : {time.time() - start} seconds")
    #     print("\n")
