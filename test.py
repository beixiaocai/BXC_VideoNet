import os
import os
import torch
import cv2
import numpy as np
import random
from model import VideoNetModel
from version import CUR_VERSION, print_version
import time


def get_frame_indices(total_frames, num_frames=16, sampling_strategy='uniform'):
    """
    根据采样策略获取帧索引
    
    Args:
        total_frames: 视频总帧数
        num_frames: 需要采样的帧数
        sampling_strategy: 采样策略
            - 'uniform': 均匀采样（默认，推荐用于测试）
            - 'random_segment': 随机片段采样
            - 'dense': 密集分段采样
            
    Returns:
        frame_indices: 长度为 num_frames 的帧索引数组
    """
    # 边界情况：视频帧数不足
    if total_frames < num_frames:
        # 重复采样直到满足 num_frames 个帧
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.extend(indices[:num_frames - len(indices)])
        return np.array(indices[:num_frames])
    
    if sampling_strategy == 'uniform':
        # 均匀采样：在整个视频范围内均匀选取 num_frames 帧
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
    elif sampling_strategy == 'random_segment':
        # 随机时间段采样：随机选择一个连续片段，然后在该片段内均匀采样 num_frames 帧
        if total_frames == num_frames:
            frame_indices = np.arange(num_frames)
        else:
            # 定义片段长度（至少覆盖视频的一半，最多覆盖整个视频）
            min_segment_length = max(num_frames, total_frames // 2)
            max_segment_length = total_frames
            
            # 随机选择片段长度
            segment_length = random.randint(min_segment_length, max_segment_length)
            
            # 随机选择起始位置
            max_start_idx = max(0, total_frames - segment_length)
            if max_start_idx > 0:
                start_idx = random.randint(0, max_start_idx)
            else:
                start_idx = 0
            
            # 在选定的片段内均匀采样 num_frames 帧
            end_idx = start_idx + segment_length - 1
            frame_indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)
        
    elif sampling_strategy == 'dense':
        # 密集采样：将视频均分成 num_frames 个片段，从每个片段中选择一帧
        segment_length = total_frames / num_frames
        frame_indices = []
        
        for i in range(num_frames):
            segment_start = int(i * segment_length)
            segment_end = int((i + 1) * segment_length)
            
            # 确保不越界
            segment_end = min(segment_end, total_frames)
            
            # 测试时选择中间帧（稳定）
            frame_idx = (segment_start + segment_end) // 2
            frame_idx = min(frame_idx, total_frames - 1)  # 防止越界
            
            frame_indices.append(frame_idx)
        
        frame_indices = np.array(frame_indices)
    
    else:
        # 默认使用均匀采样
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # 安全检查：确保返回的索引数量正确
    assert len(frame_indices) == num_frames, \
        f"采样策略 '{sampling_strategy}' 返回了 {len(frame_indices)} 个索引，期望 {num_frames} 个"
    
    # 确保所有索引都在有效范围内
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    
    return frame_indices


def predict(video_path, model_path, num_frames=16, sampling_strategy='uniform'):
    """
    视频分类预测
    
    Args:
        video_path: 视频文件路径
        model_path: 模型文件路径
        num_frames: 采样帧数（默认16）
        sampling_strategy: 采样策略
            - 'uniform': 均匀采样（默认，推荐）
            - 'random_segment': 随机片段采样
            - 'dense': 密集分段采样
    """
    print("="*60)
    print(f"采样配置: num_frames={num_frames}, strategy='{sampling_strategy}'")
    print("="*60)
    print()

    # 加载模型checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 判断checkpoint格式并提取模型权重
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # 新格式：完整checkpoint
            state_dict = checkpoint['model_state_dict']
            num_classes = checkpoint.get('num_classes', len(class_names))
            print(f"加载完整checkpoint: epoch={checkpoint.get('epoch', 'N/A')}, "
                  f"best_acc={checkpoint.get('best_acc', 'N/A'):.4f}")
        else:
            # 旧格式：直接是state_dict
            state_dict = checkpoint
            num_classes = len(class_names)
    else:
        # 非常旧的格式
        state_dict = checkpoint
        num_classes = len(class_names)
    
    # 处理多GPU训练的模型（移除'module.'前缀）
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("检测到多GPU训练模型，移除module.前缀")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 初始化模型
    model = VideoNetModel(num_classes=num_classes)
    
    # 加载模型权重
    model.load_state_dict(state_dict)
    model.eval()
    print(f"模型加载成功，类别数: {num_classes}")
    
    # 移动到适当的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Inference on device: {device}")
    
    # 预热模型 - 优化：减少首次推理的延迟
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, num_frames, 224, 224).to(device)
        model(dummy_input)
    
    # 计时开始
    start_time = time.time()
    
    # 读取视频 - 使用配置的采样策略
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video information: {total_frames} frames, {fps:.2f} FPS")

    # 根据采样策略获取帧索引
    frame_indices = get_frame_indices(total_frames, num_frames, sampling_strategy)
    print(f"Sampling strategy: '{sampling_strategy}'")
    print(f"Selected frame indices: {frame_indices.tolist()[:5]}...{frame_indices.tolist()[-2:]} (total: {len(frame_indices)} frames)")
    
    # 按照索引读取帧
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # 优化预处理：使用双线性插值加速
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()

    # 转换为模型输入 - 优化：使用更高效的数据转换
    frames = np.array(frames).transpose((3, 0, 1, 2))  # (C, T, H, W)
    frames = torch.tensor(frames).float() / 255.0
    
    # 归一化 - 优化：使用预计算的均值和标准差
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    frames = (frames - mean) / std
    
    frames = frames.unsqueeze(0).to(device)

    # 预测 - 优化：关闭梯度计算
    with torch.no_grad():
        # 允许使用TensorRT加速（如果可用）
        if device.type == 'cuda':
            # 对于CUDA设备，可以使用torch.jit.trace优化
            try:
                model = torch.jit.trace(model, frames)
            except:
                print("JIT trace not supported, using regular inference")
                pass
        
        outputs = model(frames)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

    # 计时结束
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")


    result = {
        'class': class_names[preds.item()],
        'confidence': confidence.item(),
        'all_probabilities': {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}
    }
    
    return result


if __name__ == '__main__':
    # 打印版本信息
    print_version()
    print()
    
    print("test.py start")
    
    # ==================== 配置区 ====================
    # 视频路径
    video_path = 'data/v_HorseRiding_g01_c02.avi'
    
    # 模型路径
    # model_path = "models_20251012001_ucf50/best_model_epoch242_aac0.9314.pth"
    model_path = "models_20251031001_ucf50/best_model_epoch222_aac0.9631.pth"
    
    # 采样配置
    num_frames = 16  # 采样帧数（可修改为32、64等）
    sampling_strategy = 'uniform'  # 采样策略：'uniform', 'random_segment', 'dense'
    
    # 提示：不同采样策略的使用场景
    # - 'uniform':         均匀采样，稳定可复现（推荐用于测试）
    # - 'random_segment':  随机片段采样，每次结果可能不同
    # - 'dense':           密集分段采样，适合长视频
    # ================================================

    # UCF50数据集的50个动作类别名称 （根据实际数据集修改）
    class_names = ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lunges', 'MilitaryParade', 'Mixing', 'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo']

    # 打印验证长度
    print(f"class_names: {len(class_names)}")
    print("video_path: ",video_path)

    if not os.path.exists(video_path):
        print(f"测试视频 {video_path} 不存在。请将测试视频放在当前目录并重命名为 'test_video.mp4'")
        print("或者修改代码中的 video_path 变量指向您的测试视频。")
        print("\n示例用法：")
        print(f"python {os.path.basename(__file__)}")
    else:
        result = predict(video_path, model_path, num_frames=num_frames, sampling_strategy=sampling_strategy)
        print(f'\nPredicted class: {result["class"]}')
        print(f'Confidence: {result["confidence"]:.4f}')
        print('\nTop-5 预测结果：')
        for i, (cls, prob) in enumerate(list(result['all_probabilities'].items())[:5]):
            print(f'  {i+1}. {cls}: {prob:.4f}')