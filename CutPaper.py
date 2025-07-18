import cv2
import pytesseract
import os
import re
from PIL import Image
from collections import defaultdict

# 设置Tesseract路径（根据实际安装的路径设置）
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\fhb_j\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def detect_question_numbers(image_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 图像预处理
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 使用Pytesseract获取文本位置信息
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    data = pytesseract.image_to_data(
        Image.fromarray(255 - thresh), 
        config=custom_config, 
        output_type=pytesseract.Output.DICT
    )
    
    # Collect question numbers and positions
    question_positions = []
    pattern = re.compile(r'^\d+[\.]')  # 1. 2. format
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if pattern.match(text):
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            question_positions.append((y, text.split('.')[0]))
    
    # Group questions by line
    line_groups = defaultdict(list)
    for y, num in question_positions:
        line_groups[y//10].append((y, num))  # 使用//10分组容差
    
    # 获取唯一的小题起始行
    question_starts = []
    for group in line_groups.values():
        min_y = min(y for y, _ in group)
        min_num = min(int(num) for _, num in group)
        question_starts.append((min_y, str(min_num)))
    
    # 按垂直位置排序
    question_starts.sort(key=lambda x: x[0])
    
    # 分割图像
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    height, width = image.shape[:2]
    
    prev_y = 0
    for idx, (y, num) in enumerate(question_starts):
        # 计算当前小题的结束位置（下一个小题的开始或图像底部）
        next_y = question_starts[idx+1][0] if idx+1 < len(question_starts) else height
        
        # 扩展区域确保包含完整题目
        start_y = max(0, prev_y - 10)
        end_y = min(height, next_y - 5)
        
        # 裁剪小题区域
        question_img = image[start_y:end_y, 0:width]
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{base_name}{num}.jpg")
        cv2.imwrite(output_path, question_img)
        
        prev_y = y

if __name__ == "__main__":
    input_image = "1.jpg"  # 输入试卷路径
    output_dir = "output"          # 输出目录
    
    detect_question_numbers(input_image, output_dir)