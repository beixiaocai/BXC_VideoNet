"""
BXC_VideoNet Web 管理后台
作者：北小菜
功能：视频分类模型的数据集管理、训练、测试一体化平台
"""

import os
import json
import time
import shutil
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
import numpy as np

# 导入项目模块
from model import VideoNetModel
from dataset import VideoDatasetVideo
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bxc_videonet_2025'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 最大上传

# 配置路径
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'web_data' / 'uploads'
DATASET_DIR = BASE_DIR / 'web_data' / 'dataset'
MODELS_DIR = BASE_DIR / 'web_data' / 'models'
CONFIG_FILE = BASE_DIR / 'web_data' / 'config.json'

# 创建必要目录
for dir_path in [UPLOAD_DIR, DATASET_DIR, MODELS_DIR, DATASET_DIR / 'train', DATASET_DIR / 'val']:
    dir_path.mkdir(parents=True, exist_ok=True)

# 全局变量
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_loss': 0.0,
    'train_acc': 0.0,
    'val_loss': 0.0,
    'val_acc': 0.0,
    'best_acc': 0.0,
    'progress': 0,
    'log': []
}

training_thread = None


def load_config():
    """加载配置"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'classes': [],
        'total_videos': 0,
        'train_videos': 0,
        'val_videos': 0,
        'models': []
    }


def save_config(config):
    """保存配置"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_dataset_stats():
    """获取数据集统计信息"""
    stats = {
        'classes': [],
        'total_train': 0,
        'total_val': 0
    }
    
    for phase in ['train', 'val']:
        phase_dir = DATASET_DIR / phase
        if phase_dir.exists():
            for cls_dir in phase_dir.iterdir():
                if cls_dir.is_dir():
                    videos = list(cls_dir.glob('*.*'))
                    cls_name = cls_dir.name
                    count = len(videos)
                    
                    # 查找或添加类别
                    cls_info = next((c for c in stats['classes'] if c['name'] == cls_name), None)
                    if not cls_info:
                        cls_info = {'name': cls_name, 'train': 0, 'val': 0}
                        stats['classes'].append(cls_info)
                    
                    cls_info[phase] = count
                    if phase == 'train':
                        stats['total_train'] += count
                    else:
                        stats['total_val'] += count
    
    return stats


# ==================== 路由 ====================

@app.route('/')
def index():
    """首页 - 数据集管理"""
    return render_template('index.html')


@app.route('/train_page')
def train_page():
    """训练页面"""
    return render_template('train.html')


@app.route('/test_page')
def test_page():
    """测试页面"""
    return render_template('test.html')


@app.route('/convert_page')
def convert_page():
    """模型转换页面"""
    return render_template('convert.html')


@app.route('/api/dataset/stats', methods=['GET'])
def get_stats():
    """获取数据集统计"""
    stats = get_dataset_stats()
    return jsonify(stats)


@app.route('/api/categories/list', methods=['GET'])
def list_categories():
    """获取所有类别文件夹"""
    categories = []
    
    if UPLOAD_DIR.exists():
        for cat_dir in UPLOAD_DIR.iterdir():
            if cat_dir.is_dir():
                videos = list(cat_dir.glob('*.*'))
                video_count = len([v for v in videos if v.suffix.lower() in ['.mp4', '.avi', '.mov']])
                categories.append({
                    'name': cat_dir.name,
                    'video_count': video_count
                })
    
    return jsonify({'categories': categories})


@app.route('/api/categories/create', methods=['POST'])
def create_category():
    """创建类别文件夹"""
    data = request.get_json()
    category_name = data.get('name', '').strip()
    
    if not category_name:
        return jsonify({'success': False, 'message': '请输入类别名称'})
    
    # 验证类别名称（只允许字母、数字、下划线）
    import re
    if not re.match(r'^[a-zA-Z0-9_]+$', category_name):
        return jsonify({'success': False, 'message': '类别名称只能包含字母、数字和下划线'})
    
    category_path = UPLOAD_DIR / category_name
    
    if category_path.exists():
        return jsonify({'success': False, 'message': f'类别 "{category_name}" 已存在'})
    
    try:
        category_path.mkdir(parents=True, exist_ok=True)
        return jsonify({
            'success': True,
            'message': f'类别 "{category_name}" 创建成功',
            'name': category_name
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'创建失败: {str(e)}'})


@app.route('/api/categories/delete', methods=['POST'])
def delete_category():
    """删除类别文件夹"""
    data = request.get_json()
    category_name = data.get('name', '').strip()
    
    if not category_name:
        return jsonify({'success': False, 'message': '类别名称不能为空'})
    
    category_path = UPLOAD_DIR / category_name
    
    if not category_path.exists():
        return jsonify({'success': False, 'message': f'类别 "{category_name}" 不存在'})
    
    try:
        shutil.rmtree(category_path)
        return jsonify({
            'success': True,
            'message': f'类别 "{category_name}" 已删除'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'删除失败: {str(e)}'})


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """上传视频到指定类别"""
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': '没有上传文件'})
    
    file = request.files['video']
    category_name = request.form.get('category', '').strip()
    
    if not category_name:
        return jsonify({'success': False, 'message': '请选择类别'})
    
    if file.filename == '':
        return jsonify({'success': False, 'message': '文件名为空'})
    
    # 验证文件格式
    if not file.filename or not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return jsonify({'success': False, 'message': '只支持 .mp4, .avi, .mov 格式'})
    
    # 保存到类别目录
    filename = secure_filename(file.filename or 'video.mp4')
    category_dir = UPLOAD_DIR / category_name
    
    if not category_dir.exists():
        return jsonify({'success': False, 'message': f'类别 "{category_name}" 不存在'})
    
    # 检查文件是否已存在
    filepath = category_dir / filename
    if filepath.exists():
        # 添加时间戳避免重名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        filepath = category_dir / filename
    
    try:
        file.save(str(filepath))
        return jsonify({
            'success': True,
            'message': f'上传成功: {filename}',
            'category': category_name,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})


@app.route('/api/dataset/split', methods=['POST'])
def split_dataset():
    """分割数据集"""
    data = request.get_json()
    train_ratio = float(data.get('train_ratio', 0.8))
    seed = int(data.get('seed', 42))
    
    try:
        # 清空现有数据集
        for phase in ['train', 'val']:
            phase_dir = DATASET_DIR / phase
            if phase_dir.exists():
                shutil.rmtree(phase_dir)
            phase_dir.mkdir(parents=True, exist_ok=True)
        
        # 遍历上传目录的每个类别
        np.random.seed(seed)
        total_moved = 0
        
        for class_dir in UPLOAD_DIR.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            videos = list(class_dir.glob('*.*'))
            
            if len(videos) == 0:
                continue
            
            # 打乱顺序
            videos = list(videos)
            np.random.shuffle(videos)
            
            # 计算分割点
            split_idx = int(len(videos) * train_ratio)
            train_videos = videos[:split_idx]
            val_videos = videos[split_idx:]
            
            # 创建类别目录
            (DATASET_DIR / 'train' / class_name).mkdir(parents=True, exist_ok=True)
            (DATASET_DIR / 'val' / class_name).mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            for video in train_videos:
                shutil.copy2(video, DATASET_DIR / 'train' / class_name / video.name)
            
            for video in val_videos:
                shutil.copy2(video, DATASET_DIR / 'val' / class_name / video.name)
            
            total_moved += len(videos)
        
        stats = get_dataset_stats()
        
        return jsonify({
            'success': True,
            'message': f'成功分割 {total_moved} 个视频',
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'分割失败: {str(e)}'})


@app.route('/api/dataset/clear_uploads', methods=['POST'])
def clear_uploads():
    """清空上传目录"""
    try:
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        return jsonify({'success': True, 'message': '上传目录已清空'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/models/list', methods=['GET'])
def list_models():
    """列出所有模型"""
    models = []
    
    if MODELS_DIR.exists():
        for model_file in MODELS_DIR.glob('*.pth'):
            stat = model_file.stat()
            models.append({
                'name': model_file.name,
                'size': f'{stat.st_size / 1024 / 1024:.2f} MB',
                'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                'path': str(model_file)
            })
    
    # 按创建时间倒序
    models.sort(key=lambda x: x['created'], reverse=True)
    
    return jsonify({'models': models})


@app.route('/api/train/start', methods=['POST'])
def start_training():
    """启动训练"""
    global training_thread, training_status
    
    if training_status['is_training']:
        return jsonify({'success': False, 'message': '训练正在进行中'})
    
    data = request.get_json()
    num_epochs = int(data.get('num_epochs', 50))
    batch_size = int(data.get('batch_size', 8))
    learning_rate = float(data.get('learning_rate', 0.0005))
    pretrained_model = data.get('pretrained_model', '')
    
    # 检查数据集
    stats = get_dataset_stats()
    if stats['total_train'] == 0 or stats['total_val'] == 0:
        return jsonify({'success': False, 'message': '请先上传并分割数据集'})
    
    # 重置训练状态
    training_status = {
        'is_training': True,
        'current_epoch': 0,
        'total_epochs': num_epochs,
        'train_loss': 0.0,
        'train_acc': 0.0,
        'val_loss': 0.0,
        'val_acc': 0.0,
        'best_acc': 0.0,
        'progress': 0,
        'log': []
    }
    
    # 启动训练线程
    training_thread = threading.Thread(
        target=train_model,
        args=(num_epochs, batch_size, learning_rate, pretrained_model)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'success': True, 'message': '训练已启动'})


@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """获取训练状态"""
    return jsonify(training_status)


@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    """停止训练"""
    global training_status
    
    if training_status['is_training']:
        training_status['is_training'] = False
        training_status['log'].append('用户手动停止训练')
        return jsonify({'success': True, 'message': '训练已停止'})
    
    return jsonify({'success': False, 'message': '当前没有正在进行的训练'})


@app.route('/api/test/predict', methods=['POST'])
def predict_video():
    """测试视频预测"""
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': '没有上传文件'})
    
    file = request.files['video']
    model_name = request.form.get('model_name', '')
    
    if not model_name:
        return jsonify({'success': False, 'message': '请选择模型'})
    
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return jsonify({'success': False, 'message': '模型不存在'})
    
    # 保存临时文件
    temp_video = UPLOAD_DIR / f'test_{int(time.time())}_{secure_filename(file.filename or "video.mp4")}'
    file.save(str(temp_video))
    
    try:
        # 执行推理
        result = predict_single_video(str(temp_video), str(model_path))
        
        # 删除临时文件
        if temp_video.exists():
            temp_video.unlink()
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        if temp_video.exists():
            temp_video.unlink()
        return jsonify({'success': False, 'message': f'预测失败: {str(e)}'})


def train_model(num_epochs, batch_size, learning_rate, pretrained_model):
    """训练模型（在单独线程中运行）"""
    global training_status
    
    try:
        training_status['log'].append(f'开始训练: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}')
        
        # 数据增强
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
        
        # 创建数据集
        train_dataset = VideoDatasetVideo('train', str(DATASET_DIR / 'train'), data_transforms['train'],
                                         sampling_strategy='random_segment')  # 训练时使用随机片段采样
        val_dataset = VideoDatasetVideo('val', str(DATASET_DIR / 'val'), data_transforms['val'],
                                       sampling_strategy='uniform')  # 验证时使用均匀采样
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # 初始化模型
        num_classes = len(train_dataset.classes)
        model = VideoNetModel(num_classes=num_classes)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        training_status['log'].append(f'使用设备: {device}')
        
        # 加载预训练模型
        start_epoch = 0
        if pretrained_model:
            pretrained_path = MODELS_DIR / pretrained_model
            if pretrained_path.exists():
                checkpoint = torch.load(pretrained_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                    if 'best_acc' in checkpoint:
                        training_status['best_acc'] = checkpoint['best_acc']
                else:
                    model.load_state_dict(checkpoint)
                training_status['log'].append(f'加载预训练模型: {pretrained_model}')
        
        model = model.to(device)
        
        # 优化器和损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        # 训练循环
        for epoch in range(start_epoch, num_epochs):
            if not training_status['is_training']:
                break
            
            training_status['current_epoch'] = epoch + 1
            training_status['progress'] = int((epoch + 1) / num_epochs * 100)
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                if not training_status['is_training']:
                    break
                
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
            
            training_status['train_loss'] = train_loss / train_total if train_total > 0 else 0
            training_status['train_acc'] = train_correct / train_total if train_total > 0 else 0
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if not training_status['is_training']:
                        break
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            training_status['val_loss'] = val_loss / val_total if val_total > 0 else 0
            training_status['val_acc'] = val_correct / val_total if val_total > 0 else 0
            
            scheduler.step()
            
            # 保存最佳模型
            if training_status['val_acc'] > training_status['best_acc']:
                training_status['best_acc'] = training_status['val_acc']
                model_name = f"best_model_epoch{epoch+1}_acc{training_status['val_acc']:.4f}.pth"
                model_path = MODELS_DIR / model_name
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': training_status['best_acc'],
                    'num_classes': num_classes,
                    'class_names': train_dataset.classes
                }
                torch.save(checkpoint, str(model_path))
                training_status['log'].append(f'保存最佳模型: {model_name}')
            
            # 保存最新模型
            latest_path = MODELS_DIR / 'latest_model.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': training_status['best_acc'],
                'num_classes': num_classes,
                'class_names': train_dataset.classes
            }
            torch.save(checkpoint, str(latest_path))
            
            log_msg = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {training_status['train_loss']:.4f}, Train Acc: {training_status['train_acc']:.4f}, Val Loss: {training_status['val_loss']:.4f}, Val Acc: {training_status['val_acc']:.4f}"
            training_status['log'].append(log_msg)
        
        training_status['is_training'] = False
        training_status['log'].append('训练完成！')
    
    except Exception as e:
        training_status['is_training'] = False
        training_status['log'].append(f'训练错误: {str(e)}')


def predict_single_video(video_path, model_path):
    """预测单个视频"""
    import cv2
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    if isinstance(checkpoint, dict):
        num_classes = checkpoint.get('num_classes', 50)
        class_names = checkpoint.get('class_names', [f'Class_{i}' for i in range(num_classes)])
        state_dict = checkpoint['model_state_dict']
    else:
        # 尝试从数据集获取类别
        stats = get_dataset_stats()
        class_names = [c['name'] for c in stats['classes']]
        num_classes = len(class_names)
        state_dict = checkpoint
    
    model = VideoNetModel(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames >= 16:
        frame_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
    
    cap.release()
    
    # 填充到16帧
    while len(frames) < 16:
        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    # 转换为tensor
    frames = np.array(frames[:16]).transpose((3, 0, 1, 2))
    frames = torch.tensor(frames).float() / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    frames = (frames - mean) / std
    frames = frames.unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(frames)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
    
    # 获取top5
    top5_prob, top5_idx = torch.topk(probabilities[0], min(5, len(class_names)))
    top5_results = [
        {'class': class_names[idx.item()], 'probability': prob.item()}
        for idx, prob in zip(top5_idx, top5_prob)
    ]
    
    return {
        'predicted_class': class_names[preds.item()],
        'confidence': confidence.item(),
        'top5': top5_results
    }


# ==================== 模型转换 API ====================

@app.route('/api/models/convert', methods=['POST'])
def convert_model():
    """转换模型格式"""
    data = request.get_json()
    model_name = data.get('model_name', '')
    output_format = data.get('output_format', 'onnx')  # onnx 或 openvino
    dynamic_input = data.get('dynamic_input', False)
    
    if not model_name:
        return jsonify({'success': False, 'message': '请选择模型'})
    
    model_path = MODELS_DIR / model_name
    
    if not model_path.exists():
        return jsonify({'success': False, 'message': f'模型文件不存在: {model_name}'})
    
    try:
        # 获取类别数
        stats = get_dataset_stats()
        num_classes = len(stats['classes'])
        
        if num_classes == 0:
            return jsonify({'success': False, 'message': '无法获取类别信息，请先上传数据集'})
        
        # 生成输出文件名
        base_name = model_path.stem  # 不包含后缀
        
        if output_format == 'onnx':
            output_path = MODELS_DIR / f"{base_name}.onnx"
            success = convert_to_onnx_format(model_path, output_path, num_classes, dynamic_input)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'ONNX模型转换成功',
                    'output_file': output_path.name,
                    'format': 'ONNX'
                })
            else:
                return jsonify({'success': False, 'message': 'ONNX转换失败，请查看日志'})
        
        elif output_format == 'openvino':
            # 先转换为 ONNX，再转换为 OpenVINO
            onnx_path = MODELS_DIR / f"{base_name}_temp.onnx"
            success = convert_to_onnx_format(model_path, onnx_path, num_classes, False)  # OpenVINO不需要动态输入
            
            if not success:
                return jsonify({'success': False, 'message': 'ONNX转换失败，无法继续转换为OpenVINO'})
            
            # 转换为 OpenVINO
            output_dir = MODELS_DIR / f"{base_name}_openvino"
            success = convert_to_openvino_format(onnx_path, output_dir)
            
            # 删除临时ONNX文件
            if onnx_path.exists():
                onnx_path.unlink()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'OpenVINO模型转换成功',
                    'output_dir': output_dir.name,
                    'format': 'OpenVINO'
                })
            else:
                return jsonify({'success': False, 'message': 'OpenVINO转换失败，请确保已安装 openvino-dev'})
        
        else:
            return jsonify({'success': False, 'message': f'不支持的格式: {output_format}'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'转换失败: {str(e)}'})


def convert_to_onnx_format(model_path, output_path, num_classes, dynamic_input=False):
    """转换为 ONNX 格式 - 支持完整checkpoint和纯state_dict两种格式"""
    try:
        print(f"\n开始转换为 ONNX: {model_path} -> {output_path}")
        
        # 初始化模型
        model = VideoNetModel(num_classes=num_classes)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 判断是否是完整的checkpoint（包含model_state_dict键）
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("检测到完整checkpoint，提取model_state_dict")
            state_dict = checkpoint['model_state_dict']
        else:
            # 直接是state_dict
            state_dict = checkpoint
        
        # 处理多GPU训练的模型
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("检测到多GPU训练模型，移除module.前缀")
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        
        model.eval()
        print("模型加载成功")
        
        # 创建输入张量
        batch_size = 1
        num_frames = 16
        height = 224
        width = 224
        dummy_input = torch.randn(batch_size, 3, num_frames, height, width)
        
        # 导出 ONNX
        export_kwargs = {
            'model': model,
            'args': (dummy_input,),
            'f': str(output_path),
            'export_params': True,
            'opset_version': 13,
            'do_constant_folding': True,
            'input_names': ['input'],
            'output_names': ['output'],
            'verbose': False
        }
        
        if dynamic_input:
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'frames'},
                'output': {0: 'batch_size'}
            }
            export_kwargs['dynamic_axes'] = dynamic_axes
            print("启用动态输入维度")
        
        torch.onnx.export(**export_kwargs)
        print(f"ONNX 模型已保存到: {output_path}")
        
        # 验证 ONNX 模型
        try:
            import onnx
            model_onnx = onnx.load(str(output_path))
            onnx.checker.check_model(model_onnx)
            print("ONNX 模型验证成功")
        except ImportError:
            print("未安装 onnx 库，跳过验证")
        
        return True
    
    except Exception as e:
        print(f"ONNX 转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def convert_to_openvino_format(onnx_path, output_dir):
    """转换为 OpenVINO 格式"""
    try:
        print(f"\n开始转换为 OpenVINO: {onnx_path} -> {output_dir}")
        
        # 导入 OpenVINO Model Optimizer
        try:
            import subprocess
            import sys
            
            # 创建输出目录
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取Python解释器路径（虚拟环境中的）
            python_path = sys.executable
            venv_scripts_dir = Path(python_path).parent
            mo_cmd = venv_scripts_dir / 'mo.exe' if venv_scripts_dir.exists() else 'mo'
            
            # 调用 mo 命令
            cmd = [
                str(mo_cmd),
                '--input_model', str(onnx_path),
                '--output_dir', str(output_dir)
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("OpenVINO 模型转换成功")
                print(result.stdout)
                return True
            else:
                print(f"OpenVINO 转换失败: {result.stderr}")
                return False
        
        except FileNotFoundError as e:
            print(f"错误: 未找到 'mo' 命令: {e}")
            print("请确认已安装 openvino-dev")
            print("安装命令: pip install openvino-dev")
            return False
    
    except Exception as e:
        print(f"OpenVINO 转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("BXC_VideoNet Web 管理后台")
    print("=" * 60)
    print("启动服务器...")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
