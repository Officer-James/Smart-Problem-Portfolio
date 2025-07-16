import cv2
import pytesseract
import os
import re
from PIL import Image

# 设置Tesseract路径（根据实际安装的路径设置）
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\fhb_j\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def split_question(image_path, output_dir='output'):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化处理：自适应阈值
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 使用 pytesseract 识别文本（识别语言为中文和英文）
    text = pytesseract.image_to_string(thresh, lang='chi_sim+eng', config='--psm 6')

    # 使用正则表达式匹配题号（如 1.、2. 等）
    pattern = r'\d+\.'  # 匹配数字和点（例如 "1."、"2." 等）
    question_positions = [match.start() for match in re.finditer(pattern, text)]

    # 添加结束位置（文本的末尾）
    question_positions.append(len(text))

    # 计算每一道题目在图片中的位置
    for i in range(len(question_positions) - 1):
        question_text = text[question_positions[i]:question_positions[i + 1]].strip()
        if question_text:  # 如果该部分不是空文本
            # 获取题目在图片中的位置
            start_pos = question_positions[i]
            end_pos = question_positions[i + 1]

            # 找到题号的上下文区域，进一步切分出每一道题
            x1, y1, w, h = pytesseract.image_to_boxes(thresh)[start_pos:end_pos]  # 提取该题文本的位置
            x2, y2 = x1 + w, y1 + h

            # 裁剪出题目区域
            question_img = img[y1:y2, x1:x2]

            # 构造输出文件路径
            file_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i + 1}.jpg"
            output_path = os.path.join(output_dir, file_name)

            # 保存切分出来的题目图片
            cv2.imwrite(output_path, question_img)

            print(f"保存题目 {i + 1} 到 {output_path}")

# 示例用法
image_path = 'test.jpeg'  # 替换为你的数学试卷图片路径
split_question(image_path)
