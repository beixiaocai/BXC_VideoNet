import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class VideoDatasetVideo(Dataset):
    def __init__(self, phase, root_dir, transform=None, num_frames=16, target_size=(224, 224), 
                 sampling_strategy='random_segment'):
        """
        视频数据集加载器
        
        Args:
            phase: 'train' 或 'val'
            root_dir: 数据集根目录
            transform: 数据增强变换
            num_frames: 采样帧数，默认16
            target_size: 目标图像尺寸
            sampling_strategy: 采样策略
                - 'uniform': 均匀采样（验证集推荐）
                - 'random_segment': 随机时间段采样（训练集推荐，默认）
                - 'dense': 密集采样整个视频
        """
        self.phase = phase
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.target_size = target_size
        self.sampling_strategy = sampling_strategy
        
        # 训练阶段默认使用随机采样，验证阶段使用均匀采样
        if self.sampling_strategy == 'random_segment' and phase == 'val':
            self.sampling_strategy = 'uniform'
        
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for video_file in os.listdir(class_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(class_path, video_file), self.class_to_idx[class_name]))

        print(f"VideoDatasetVideo.__init__()  sampling_strategy:{sampling_strategy}, phase:{self.phase}, root_dir:{root_dir}, samples:{len(self.samples)}, "
              f"classes:{self.classes}, sampling_strategy:{self.sampling_strategy}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []

        # 读取视频帧
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 根据采样策略获取帧索引
        frame_indices = self._get_frame_indices(total_frames)
        
        # 调试日志（可选，用于验证采样是否正确）
        if idx < 3 and self.phase == 'train':  # 只打印前3个样本
            print(f"[{self.phase}] Video: {os.path.basename(video_path)}, "
                  f"Total frames: {total_frames}, "
                  f"Sampling strategy: {self.sampling_strategy}, "
                  f"Selected indices: {frame_indices.tolist()[:5]}...{frame_indices.tolist()[-2:]} "
                  f"(total: {len(frame_indices)} frames)")

        # 按照索引读取帧
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
            blank_frame = Image.new('RGB', self.target_size, color=0)
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
    
    def _get_frame_indices(self, total_frames):
        """
        根据采样策略获取帧索引，始终返回 num_frames 个索引
        
        Args:
            total_frames: 视频总帧数
            
        Returns:
            frame_indices: 长度为 num_frames 的帧索引数组
        """
        # 边界情况：视频帧数不足
        if total_frames < self.num_frames:
            # 重复采样直到满足 num_frames 个帧
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.extend(indices[:self.num_frames - len(indices)])
            return np.array(indices[:self.num_frames])
        
        if self.sampling_strategy == 'uniform':
            # 均匀采样：在整个视频范围内均匀选取 num_frames 帧
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
        elif self.sampling_strategy == 'random_segment':
            # 随机时间段采样：随机选择一个连续片段，然后在该片段内均匀采样 num_frames 帧
            if total_frames == self.num_frames:
                # 视频长度正好等于需要的帧数
                frame_indices = np.arange(self.num_frames)
            else:
                # 定义片段长度（至少覆盖视频的一半，最多覆盖整个视频）
                min_segment_length = max(self.num_frames, total_frames // 2)
                max_segment_length = total_frames
                
                # 随机选择片段长度
                if self.phase == 'train':
                    segment_length = random.randint(min_segment_length, max_segment_length)
                else:
                    # 验证时使用固定的片段长度（整个视频）
                    segment_length = total_frames
                
                # 随机选择起始位置
                max_start_idx = max(0, total_frames - segment_length)
                if self.phase == 'train' and max_start_idx > 0:
                    start_idx = random.randint(0, max_start_idx)
                else:
                    start_idx = 0
                
                # 在选定的片段内均匀采样 num_frames 帧
                end_idx = start_idx + segment_length - 1
                frame_indices = np.linspace(start_idx, end_idx, self.num_frames, dtype=int)
            
        elif self.sampling_strategy == 'dense':
            # 密集采样：将视频均分成 num_frames 个片段，从每个片段中选择一帧
            segment_length = total_frames / self.num_frames
            frame_indices = []
            
            for i in range(self.num_frames):
                segment_start = int(i * segment_length)
                segment_end = int((i + 1) * segment_length)
                
                # 确保不越界
                segment_end = min(segment_end, total_frames)
                
                # 从每个片段中选择一帧
                if self.phase == 'train' and segment_end > segment_start:
                    # 训练时：随机选择
                    frame_idx = random.randint(segment_start, segment_end - 1)
                else:
                    # 验证时：选择中间帧
                    frame_idx = (segment_start + segment_end) // 2
                    frame_idx = min(frame_idx, total_frames - 1)  # 防止越界
                
                frame_indices.append(frame_idx)
            
            frame_indices = np.array(frame_indices)
        
        else:
            # 默认使用均匀采样
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        # 安全检查：确保返回的索引数量正确
        assert len(frame_indices) == self.num_frames, \
            f"采样策略 '{self.sampling_strategy}' 返回了 {len(frame_indices)} 个索引，期望 {self.num_frames} 个"
        
        # 确保所有索引都在有效范围内
        frame_indices = np.clip(frame_indices, 0, total_frames - 1)
        
        return frame_indices
class VideoDatasetImage(Dataset):
    def __init__(self, phase, root_dir, transform=None, num_frames=16, target_size=(224, 224)):
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
                            image_names = [f for f in os.listdir(video_dir_path) if
                                           f.endswith(('.jpg', '.jpeg', '.png'))]
                            if len(image_names) > 0:
                                self.samples.append((video_dir_path, self.class_to_idx[class_name]))

        print(
            f"dataset_image.__init__()  phase:{self.phase}, root_dir:{root_dir}, samples:{len(self.samples)}, classes:{self.classes}")

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
            blank_frame = Image.new('RGB', self.target_size, color=0)
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
            return Image.new('RGB', self.target_size, color=0)