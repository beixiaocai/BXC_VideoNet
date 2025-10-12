import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random


class VideoDataset(Dataset):
    def __init__(self, phase,root_dir, transform=None, num_frames=16, target_size=(224, 224)):
        self.phase = phase
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.target_size = target_size
        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        # 安全遍历类别文件夹
        if os.path.exists(root_dir):
            # 获取类别列表并过滤掉非文件夹项
            self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            # 遍历每个类别文件夹下的视频文件夹
            for class_name in self.classes:
                class_path = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                for video_dir in os.listdir(class_path):
                    if video_dir.startswith('v_'):  # 确保只处理视频文件夹
                        video_dir_path = os.path.join(class_path, video_dir)
                        if os.path.isdir(video_dir_path):  # 确保是文件夹
                            # 检查文件夹中是否有足够的图片
                            image_names = [f for f in os.listdir(video_dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                            if len(image_names) > 0:
                                self.samples.append((video_dir_path, self.class_to_idx[class_name]))

        print(f"dataset_image.__init__()  phase:{self.phase}, root_dir:{root_dir}, samples:{len(self.samples)}, classes:{self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir_path, label = self.samples[idx]

        # 获取所有图片并排序
        image_names = [f for f in os.listdir(video_dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_names.sort()  # 确保顺序一致

        # 优化采样策略：均匀采样而不是只取前N帧
        frames = []
        if len(image_names) > 0:
            # 均匀采样num_frames帧
            if len(image_names) >= self.num_frames:
                # 计算采样间隔
                step = max(1, len(image_names) // self.num_frames)
                # 使用随机起点以增加多样性
                start_idx = random.randint(0, max(0, len(image_names) - self.num_frames * step))
                selected_indices = [start_idx + i * step for i in range(self.num_frames)]
                # 确保索引不越界
                selected_indices = [min(idx, len(image_names) - 1) for idx in selected_indices]

                for i in selected_indices:
                    image_name = image_names[i]
                    image_path = os.path.join(video_dir_path, image_name)
                    frame = self._load_image(image_path)
                    frames.append(frame)
            else:
                # 如果帧数不足，先加载所有帧，然后复制填充
                for image_name in image_names:
                    image_path = os.path.join(video_dir_path, image_name)
                    frame = self._load_image(image_path)
                    frames.append(frame)

                # 复制填充到所需帧数
                while len(frames) < self.num_frames:
                    # 随机选择一帧进行复制
                    frames.append(frames[random.randint(0, len(frames) - 1)].copy())

        # 确保至少有num_frames帧
        if len(frames) < self.num_frames:
            # 创建空白帧进行填充
            blank_frame = Image.new('RGB', self.target_size, (0, 0, 0))
            while len(frames) < self.num_frames:
                frames.append(blank_frame.copy())

        # 使用同一个随机变换参数处理同一视频的所有帧，保持一致性
        if self.transform is not None:
            # 使用同一个随机变换参数处理同一视频的所有帧，保持一致性
            seed = np.random.randint(2147483647)
            transformed_frames = []
            for frame in frames:
                random.seed(seed)
                torch.manual_seed(seed)
                transformed_frames.append(self.transform(frame))
            frames = transformed_frames
        else:
            # 如果没有transform，使用ToTensor
            to_tensor = transforms.ToTensor()
            frames = [to_tensor(frame) for frame in frames]

        # 转换为tensor并调整维度顺序
        frames = torch.stack(frames)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

        return frames, torch.tensor(label)

    def _load_image(self, image_path):
        """安全加载图像并进行预处理"""
        try:
            frame = cv2.imread(image_path)
            
            if frame is None:
                # 如果图片读取失败，用黑色帧填充
                frame = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            else:
                # 转换为RGB (OpenCV是BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 使用双线性插值调整大小，平衡质量和速度
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # 转换为PIL Image
            return Image.fromarray(frame)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # 返回空白图像
            return Image.new('RGB', self.target_size, (0, 0, 0))