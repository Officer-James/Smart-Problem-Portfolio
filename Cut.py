import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

# 设置Tesseract路径（根据实际安装位置修改）
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\fhb_j\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def split_math_questions(image_path):
    # 1. 读取并预处理图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # 2. 检测文本区域轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. 提取并排序文本块（从上到下）
    text_blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 15:  # 过滤小噪点
            text_blocks.append((x, y, w, h))
    
    text_blocks.sort(key=lambda b: b[1])  # 按y坐标排序
    
    # 4. Read numbers and divide
    questions = []
    current_question = []
    question_num = 0
    
    for i, (x, y, w, h) in enumerate(text_blocks):
        block_img = img[y:y+h, x:x+w]
        text = pytesseract.image_to_string(block_img, config='--psm 6 --oem 3', lang='chi_sim+eng')
        
        # Match numbers(1.2.format)
        if re.search(r'^\s*(\d+)[\.]', text.strip()):
            if current_question:  # Save the previous question
                questions.append(merge_blocks(current_question))
                current_question = []
            question_num += 1
        current_question.append((x, y, w, h))
        
    if current_question:  # Add the last question
        questions.append(merge_blocks(current_question))
   
    # 5. Visualize and save results
    output_img = img.copy()
    for i, (x, y, w, h) in enumerate(questions):
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(f"output/23uestion_{i+1}.jpg", img[y:y+h, x:x+w])
    
    return output_img, questions

def merge_blocks(blocks):
    """合并属于同一小题的多个文本块"""
    xs = [x for x,_,_,_ in blocks]
    ys = [y for _,y,_,_ in blocks]
    xe = [x+w for x,_,w,_ in blocks]
    ye = [y+h for _,y,_,h in blocks]
    return (min(xs), min(ys), max(xe)-min(xs), max(ye)-min(ys))

# 使用示例
output_img, questions = split_math_questions("1.jpg")
cv2.imwrite("output/output.jpg", output_img)
print(f"切分到 {len(questions)} 道小题")