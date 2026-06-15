import cv2
import os
import numpy as np

def draw_mot_boxes(image_folder, txt_path, output_folder, img_format='%06d.jpg'):
    """
    将MOT跟踪结果的边界框绘制到图像上
    参数:
        image_folder: 存放原始图像的文件夹路径
        txt_path: MOT结果文件路径（每行格式：帧号,ID,x1,y1,宽度,高度,...）
        output_folder: 输出图像保存路径
        img_format: 图像文件名格式（如%06d.jpg表示000001.jpg）
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载跟踪结果数据
    with open(txt_path, 'r') as f:
        tracks = f.readlines()
    
    # 按帧号分组跟踪结果
    frame_data = {}
    for track in tracks:
        data = track.strip().split(',')
        frame_id = int(data[0])
        obj_id = int(float(data[1]))
        x1, y1, w, h = map(float, data[2:6])  # 转换为浮点数
        
        if frame_id not in frame_data:
            frame_data[frame_id] = []
        frame_data[frame_id].append((obj_id, x1, y1, w, h))
    
    # 处理每一帧图像
    for frame_id, boxes in frame_data.items():
        img_path = os.path.join(image_folder, img_format % frame_id)
        if not os.path.exists(img_path):
            print(f"警告：图像 {img_path} 不存在，跳过")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {img_path}，跳过")
            continue
        
        # 绘制每个目标的框和ID
        for obj_id, x1, y1, w, h in boxes:
            x2, y2 = int(x1 + w), int(y1 + h)
            x1, y1 = int(x1), int(y1)
            
            # 绘制矩形框（绿色边框，厚度2）
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制ID标签（蓝色背景+白色文字）
            label = f"ID:{obj_id}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (255, 0, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 保存结果图像
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, img)
        print(f"已处理帧 {frame_id} -> 保存至 {output_path}")

if __name__ == "__main__":
    # ====== 用户需修改以下参数 ======
    IMAGE_FOLDER = "/home/LiaoYangHao/LinMingRun/CenterTrack/data/mot17/test/seq3/img1"   # 原始图像目录
    TXT_PATH = "/home/LiaoYangHao/LinMingRun/FairMOT/dataset/MultiFishDataSet/images/results/MOT17_test_public_dla34/seq3.txt"  # MOT结果文件
    OUTPUT_FOLDER = "/home/LiaoYangHao/LinMingRun/CenterTrack/img/FairMOT"    # 输出目录
    # ==============================
    
    draw_mot_boxes(IMAGE_FOLDER, TXT_PATH, OUTPUT_FOLDER)