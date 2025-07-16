import os
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_jpg(pdf_path, output_dir, dpi=200, fmt='jpeg'):
    """
    将PDF的每一页转换为JPG图片
    :param pdf_path: PDF文件路径
    :param output_dir: 输出目录
    :param dpi: 输出图像分辨率（默认200）
    :param fmt: 输出格式（默认jpg）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.isfile(pdf_path):
        print(f"错误：文件 {pdf_path} 不存在")
        return
    
    # 获取PDF文件名（不含扩展名）
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    print(f"开始转换: {pdf_name}.pdf -> {fmt}图片序列...")
    
    try:
        # 使用pdf2image转换PDF为图像列表
        
        if os.name == 'nt':
            poppler_path = r"C:\Development\poppler-24.08.0\Library\bin"  # 替换为您的实际路径
            
        images = convert_from_path(
            pdf_path, 
            dpi=dpi, 
            fmt=fmt,
            poppler_path=poppler_path  # 指定 Poppler 路径
        )
        
        # 保存所有图像
        for i, image in enumerate(images, start=1):
            # 创建文件名（纯数字序列）
            output_path = os.path.join(output_dir, f"{i}.{fmt}")
            
            # 保存图像
            image.save(output_path, fmt.upper())
            print(f"已保存: {output_path}")
        
        print(f"转换完成! 共转换 {len(images)} 页")
        
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")

if __name__ == "__main__":
    # 配置参数
    pdf_file = "src/data/pdf/24FDSX.pdf"       # 替换为你的PDF文件路径
    output_folder = "src/data/jpg/24FDSX"   # 输出目录
    resolution = 300               # DPI分辨率（推荐300用于打印/高质量）
    
    # 执行转换
    pdf_to_jpg(pdf_file, output_folder, dpi=resolution)