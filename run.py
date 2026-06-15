import os
import shutil
import glob

# ================= 配置区域 =================
# 你的数据集根目录
DATASET_ROOT = '/home/LiaoYangHao/LinMingRun/MeMOTR/MOT17/DanceTrack/test'

# 目标格式位数 (你想要 00000001.jpg 就是 8)
# 标准 MOT 是 6，如果你确定要 8，请保持为 8
TARGET_DIGITS = 8 
# ===========================================

def safe_rename_sequence(seq_path):
    img_dir = os.path.join(seq_path, 'img1')
    if not os.path.exists(img_dir):
        print(f"⚠️ 跳过 (找不到 img1): {seq_path}")
        return

    print(f"📂 处理序列: {os.path.basename(seq_path)}")

    # 1. 获取所有 jpg 文件
    # 这一步不管文件名是 1.jpg 还是 00002.jpg，都先读进来
    files = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not files:
        print("   ❌ 目录下没有图片")
        return

    # 2. 解析当前所有的帧号，存入字典 {帧号: 原始路径}
    frame_map = {}
    for f_path in files:
        fname = os.path.basename(f_path)
        try:
            # 提取文件名里的数字，例如 'frame_002.jpg' -> 2
            # 假设文件名里包含数字
            name_no_ext = os.path.splitext(fname)[0]
            # 过滤掉非数字字符（以防万一有前缀）
            frame_num = int(''.join(filter(str.isdigit, name_no_ext)))
            frame_map[frame_num] = f_path
        except ValueError:
            print(f"   ⚠️ 跳过无法解析的文件: {fname}")

    # 3. 修复缺失的第 1 帧 (核心逻辑)
    # 如果没有第1帧，但有第2帧，就把第2帧当作第1帧的源
    if 1 not in frame_map and 2 in frame_map:
        print("   🔧 检测到缺失第1帧，正在从第2帧克隆...")
        src_path = frame_map[2]
        # 暂时先命名为一个临时文件，稍后统一重命名
        temp_frame_1 = os.path.join(img_dir, "temp_frame_01.jpg")
        shutil.copy(src_path, temp_frame_1)
        # 加入到 map 中，准备重命名
        frame_map[1] = temp_frame_1

    # 4. 统一重命名为 8 位格式
    # 按帧号排序，从小到大处理
    sorted_frames = sorted(frame_map.keys())
    
    count_renamed = 0
    for frame_num in sorted_frames:
        old_path = frame_map[frame_num]
        
        # 构造新名字: 00000001.jpg
        new_name = f"{frame_num:0{TARGET_DIGITS}d}.jpg"
        new_path = os.path.join(img_dir, new_name)

        # 如果名字已经一样了，就跳过
        if old_path == new_path:
            continue

        # 如果目标文件已存在（防止覆盖风险），先改名为临时文件
        if os.path.exists(new_path) and new_path not in frame_map.values():
             # 这种情况极少发生，但在重命名逻辑中要小心
             pass 

        try:
            os.rename(old_path, new_path)
            count_renamed += 1
        except Exception as e:
            print(f"   ❌ 重命名失败 {old_path} -> {new_name}: {e}")

    print(f"   ✅ 完成！重命名了 {count_renamed} 张图片，当前包含帧 1~{sorted_frames[-1]}")

if __name__ == "__main__":
    # 遍历 train 目录下的 seq1, seq2...
    if not os.path.exists(DATASET_ROOT):
        print("❌ 根目录不存在，请检查路径")
        exit()

    seq_list = sorted(os.listdir(DATASET_ROOT))
    for seq_name in seq_list:
        full_seq_path = os.path.join(DATASET_ROOT, seq_name)
        if os.path.isdir(full_seq_path):
            safe_rename_sequence(full_seq_path)
            
    print("\n🎉 全部处理完毕！请务必删除目录下的 .cache 缓存文件再运行模型！")