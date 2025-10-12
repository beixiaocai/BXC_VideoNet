import os
import torch
import cv2
import numpy as np
from model import XcVideoNetModel
import time


def predict(video_path, model_path):

    # 初始化模型
    model = XcVideoNetModel(num_classes=len(class_names))
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError:
        # 如果是多GPU训练的模型，需要处理
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # 移除DataParallel包装的键前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    model.eval()
    
    # 移动到适当的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Inference on device: {device}")
    
    # 预热模型 - 优化：减少首次推理的延迟
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 16, 224, 224).to(device)
        model(dummy_input)
    
    # 计时开始
    start_time = time.time()
    
    # 读取视频 - 优化：使用更高效的视频读取方式
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video information: {total_frames} frames, {fps:.2f} FPS")

    # 优化采样策略：如果视频很长，可以采用滑动窗口采样
    if total_frames >= 16:
        # 均匀采样16帧
        frame_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
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
    else:
        # 填充帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(16):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
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
    print("test.py start")
    video_path = 'data/v_HulaHoop_g01_c01.avi'
    model_path = "models/best_model_epoch16_aac0.9778.pth"
    class_names = ['Basketball', 'BasketballDunk', 'Biking']  # 根据实际数据集修改

    if not os.path.exists(video_path):
        print(f"测试视频 {video_path} 不存在。请将测试视频放在当前目录并重命名为 'test_video.mp4'")
        print("或者修改代码中的 video_path 变量指向您的测试视频。")
        print("\n示例用法：")
        print(f"python {os.path.basename(__file__)}")
    else:
        result = predict(video_path,model_path)
        print(f'Predicted class: {result["class"]}')
        print(f'Confidence: {result["confidence"]:.4f}')
        print('Class probabilities:')
        for cls, prob in result['all_probabilities'].items():
            print(f'  {cls}: {prob:.4f}')