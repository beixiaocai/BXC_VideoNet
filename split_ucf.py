import os
import shutil
import random
from pathlib import Path


def split_dataset(src_dir,train_dir,val_dir,split_ratio=0.8, seed=42):
    """
    将 UCF50 数据集按类别分层划分为训练集和验证集

    Args:
        src_dir (str or Path): 原始数据集根目录
        train_dir (str or Path): 训练集输出目录
        val_dir (str or Path): 验证集输出目录
        split_ratio (float): 训练集占比 (0 < ratio < 1)
        seed (int): 随机种子，保证可复现
    """
    src_path = Path(src_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)

    # 设置随机种子
    random.seed(seed)

    # 获取所有类别（子文件夹）
    class_dirs = [f for f in src_path.iterdir() if f.is_dir()]
    class_dirs.sort()  # 保证顺序一致

    print(f"找到 {len(class_dirs)} 个动作类别：")
    for cls_dir in class_dirs:
        print(f"  - {cls_dir.name}")

    # 创建输出目录
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        videos = list(cls_dir.glob('*.*'))  # 获取所有视频文件
        videos.sort()
        random.shuffle(videos)

        # 计算划分数量
        num_train = int(len(videos) * split_ratio)
        train_videos = videos[:num_train]
        val_videos = videos[num_train:]

        # 创建类别子目录
        (train_path / cls_name).mkdir(exist_ok=True)
        (val_path / cls_name).mkdir(exist_ok=True)

        # 复制文件
        for video in train_videos:
            shutil.copy(video, train_path / cls_name / video.name)

        for video in val_videos:
            shutil.copy(video, val_path / cls_name / video.name)

        print(f"类别 '{cls_name}': {len(train_videos)} 训练, {len(val_videos)} 验证")

    print(f"\n数据集划分完成！")
    print(f"训练集路径: {train_path}")
    print(f"验证集路径: {val_path}")


# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    # 修改以下路径为你自己的路径
    SRC_DIR = "E:\\download\\UCF50"  # 你的原始 UCF50 文件夹
    TRAIN_DIR = "E:\\download\\BXC_VideoNet\\UCF50\\train"  # 输出训练集路径
    VAL_DIR = "E:\\download\\BXC_VideoNet\\UCF50\\val"  # 输出验证集路径
    SPLIT_RATIO = 0.8  # 80% 训练，20% 验证

    split_dataset(
        src_dir=SRC_DIR,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        split_ratio=SPLIT_RATIO,
        seed=42
    )