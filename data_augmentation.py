import os
import cv2
import numpy as np
import random

def augment_image(image):
    """返回 5 个增强版本"""
    augmented_images = []

    h, w = image.shape[:2]

    for _ in range(3):
        aug = image.copy()

        # 随机选择一个增强组合
        if random.random() < 0.5:
            # 随机旋转 ±15°
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            aug = cv2.warpAffine(aug, M, (w, h))
        if random.random() < 0.5:
            # 改变亮度
            factor = random.uniform(0.7, 1.4)
            aug = np.clip(aug * factor, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            # 模糊
            k = random.choice([3, 5])
            aug = cv2.GaussianBlur(aug, (k, k), 0)

        augmented_images.append(aug)

    return augmented_images

def process_folder(folder_path):
    valid_exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in valid_exts]
    
    idx = len(files) + 1  # 起始编号为原图数量 + 1

    for filename in files:
        path = os.path.join(folder_path, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"读取失败: {path}")
            continue

        aug_imgs = augment_image(img)

        for aug in aug_imgs:
            save_path = os.path.join(folder_path, f"{idx}.jpg")
            cv2.imwrite(save_path, aug)
            print(f"Saved: {save_path}")
            idx += 1

if __name__ == "__main__":
    base_dir = "dataset"
    for subfolder in ["wrong", "right"]:
        full_path = os.path.join(base_dir, subfolder)
        if os.path.exists(full_path):
            process_folder(full_path)
        else:
            print(f"未找到文件夹: {full_path}")
