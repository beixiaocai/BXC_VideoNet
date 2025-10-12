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
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for video_file in os.listdir(class_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(class_path, video_file), self.class_to_idx[class_name]))

        print(f"dataset_video.__init__()  phase:{self.phase}, root_dir:{root_dir}, samples:{len(self.samples)}, classes:{self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        # 读取视频帧
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                # 填充黑色帧
                frame = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)

            frame_pil = Image.fromarray(frame)
            frames.append(frame_pil)

        cap.release()

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

        # frames = np.array(frames)  # (T, H, W, C)
        # frames = frames.transpose((3, 0, 1, 2))  # (C, T, H, W)
        # frames = frames.astype(np.float32) / 255.0
        #
        # if self.transform:
        #     frames = self.transform(frames)
        #
        # return torch.tensor(frames), torch.tensor(label)