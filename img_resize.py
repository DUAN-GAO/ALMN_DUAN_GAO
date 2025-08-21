import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def resize_with_padding(image_path, output_path, target_size=(640, 640), color=(114, 114, 114)):
    """
    保持宽高比调整图片尺寸，用指定颜色填充多余区域
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        target_size: 目标尺寸 (width, height)
        color: 填充颜色 (B, G, R)
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    original_h, original_w = img.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    
    # 调整图片尺寸
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建目标画布
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    # 计算放置位置（居中）
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # 将调整后的图片放到画布上
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    # 保存图片
    cv2.imwrite(output_path, canvas)
    
    return (x_offset, y_offset, scale)  # 返回填充信息，用于调整标注框

def process_directory(input_dir, output_dir, target_size=(640, 640)):
    """
    处理整个目录下的图片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_dir) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"找到 {len(image_files)} 张图片")
    
    for image_file in tqdm(image_files, desc="处理图片"):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        resize_with_padding(input_path, output_path, target_size)

# 使用示例
if __name__ == "__main__":
    input_folder = "./data/"
    output_folder = "./train_data/"
    target_size = (640, 640)  # YOLOv5常用尺寸
    
    process_directory(input_folder, output_folder, target_size)
    print("图片尺寸统一完成！")