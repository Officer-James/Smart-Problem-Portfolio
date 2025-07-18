from openai import OpenAI

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

# 设置Tesseract路径（根据实际安装位置修改）
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\fhb_j\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def classify(img_path):
    img = cv2.imread(img_path)
    #Use Tesseract to extract text from image, may contain Chinese characters
    text = pytesseract.image_to_string(img, config='--psm 6 --oem 3', lang='chi_sim+eng')
    
    print(text)
    
    client = OpenAI(api_key="sk-6f6f21ed49644ad5b5f5aec4c3734a96", base_url="https://api.deepseek.com")

    response_subject = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "你是一个智能学习助手，会对任何一个问题判断它属于哪个科目的什么知识点，只需要输出科目即可，将输出翻译成英文，并且只输出英文结果"},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    
    response_knoledge = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "你是一个智能学习助手，会对任何一个问题判断它属于哪个科目的什么知识点，只需要输出知识点即可，将输出翻译成英文，并且只输出英文结果"},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    
    return response_subject.choices[0].message.content, response_knoledge.choices[0].message.content

if __name__ == '__main__':
    img_path = '2.jpg'
    subject, knoledge = classify(img_path)
    print(subject)
    print(knoledge)