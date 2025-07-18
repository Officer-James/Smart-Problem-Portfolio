import os

def rename_images_in_folder(folder_path):
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in valid_extensions]

    files.sort()  # 可选：按文件名排序
    for idx, filename in enumerate(files, 1):
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{idx}{ext}"
        if filename == new_name:
            continue  # 如果文件名已是目标名，跳过
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        if os.path.exists(dst):
            os.remove(dst)  # 防止目标文件名已存在
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")
    
    print(f"\nTotal files in '{os.path.basename(folder_path)}': {len(files)}\n")

if __name__ == "__main__":
    base_dir = "dataset"
    wrong_path = os.path.join(base_dir, "wrong")
    right_path = os.path.join(base_dir, "right")

    if os.path.isdir(wrong_path):
        rename_images_in_folder(wrong_path)
    else:
        print(f"Folder not found: {wrong_path}")

    if os.path.isdir(right_path):
        rename_images_in_folder(right_path)
    else:
        print(f"Folder not found: {right_path}")
