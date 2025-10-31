import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# from dataset_image import VideoDataset
from dataset import VideoDatasetVideo
from model import VideoNetModel
from version import CUR_VERSION, print_version
from tqdm import tqdm
import numpy as np
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        """保存模型当验证损失减少时"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 由于EarlyStopping类无法直接访问optimizer和scheduler等，
        # 这里我们只保存模型状态，但在外部的主训练循环中会保存完整的检查点
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# 设置随机种子以确保可重复性
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 初始化随机种子
set_random_seed(42)


def train():
    print("train() start")
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)

    # 数据增强 - 调整数据增强策略，更适合轻量级模型
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
    image_datasets = {
        'train': VideoDatasetVideo('train', train_dir, data_transforms['train'], 
                                   sampling_strategy='random_segment'),  # 训练时使用随机片段采样
        'val': VideoDatasetVideo('val', val_dir, data_transforms['val'],
                                 sampling_strategy='uniform')  # 验证时使用均匀采样
    }

    # 数据加载器 - 根据GPU内存调整批次大小

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, 
                      num_workers=4, pin_memory=True)  # 启用pin_memory加速
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # 模型初始化
    model = VideoNetModel(num_classes=len(class_names))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = model.to(device)
    
    # 使用DataParallel来利用多个GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    # 损失函数和优化器 - 调整参数以适合轻量级模型
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑减少过拟合
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)  # 调整学习率和权重衰减
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=300, eta_min=1e-6
    )

    # 记录训练历史
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # 加载预训练模型（如果提供了路径）
    start_epoch = 0
    best_acc = 0.0
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        
        # 加载检查点
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        
        # 检查是否是完整的检查点（包含模型、优化器和调度器状态）
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 尝试加载优化器状态（如果存在）
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state")
            
            # 尝试加载调度器状态（如果存在）
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Loaded scheduler state")
            
            # 尝试加载训练历史和最佳准确率（如果存在）
            if 'train_loss_history' in checkpoint:
                train_loss_history = checkpoint['train_loss_history']
            if 'val_loss_history' in checkpoint:
                val_loss_history = checkpoint['val_loss_history']
            if 'train_acc_history' in checkpoint:
                train_acc_history = checkpoint['train_acc_history']
            if 'val_acc_history' in checkpoint:
                val_acc_history = checkpoint['val_acc_history']
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        else:
            # 仅加载模型权重
            model.load_state_dict(checkpoint)
            print("Loaded only model weights")
        
        print(f"Successfully loaded pretrained model. Starting training from epoch {start_epoch}")
    else:
        print("No pretrained model provided or file not found. Starting from scratch.")

    # 早停机制
    early_stopping = EarlyStopping(patience=50, verbose=True)  # 调整早停耐心值

    # 训练循环

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs} started...')
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # 使用tqdm显示进度条
            with tqdm(total=len(dataloaders[phase]), desc=f'{phase}') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            # 使用梯度裁剪防止梯度爆炸
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': running_loss / ((pbar.n + 1) * inputs.size(0)),
                        'acc': running_corrects.double() / ((pbar.n + 1) * inputs.size(0))
                    })
                    pbar.update(1)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 记录历史数据
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
                # 在训练阶段结束后更新学习率
                scheduler.step()
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 在验证阶段检查早停
            if phase == 'val':
                last_model_path = os.path.join(models_save_dir, 'last_model.pth')
                # 保存完整的检查点，包括模型、优化器、调度器状态等
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'train_acc_history': train_acc_history,
                    'val_acc_history': val_acc_history,
                    'best_acc': best_acc
                }
                torch.save(checkpoint, last_model_path)
                early_stopping(epoch_loss, model, last_model_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    # 加载最佳模型权重
                    model.load_state_dict(torch.load(last_model_path))
                    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

        if phase == 'val' and epoch_acc > best_acc:
            best_model_path = os.path.join(models_save_dir, f'best_model_epoch{epoch}_aac{epoch_acc:.4f}.pth')
            best_acc = epoch_acc
            # 保存完整的检查点
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
                'best_acc': best_acc
            }
            torch.save(best_checkpoint, best_model_path)
            print(f'Best model saved: {best_model_path}')

    print(f'Best accuracy: {best_acc:.4f}')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'final_model.pth')
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


if __name__ == '__main__':
    # 打印版本信息
    print_version()
    print()
    
    pretrained_model_path = None # 预训练模型路径，不填写则不加载预训练模型
    # pretrained_model_path = "models_20251012_ucf50/best_model_epoch242_aac0.9314.pth"
    data_dir = 'E:\\download\\BXC_VideoNet\\UCF50'
    models_save_dir = "models"
    num_epochs = 300 # 训练总周期
    batch_size = 16  # 批次大小（轻量级模型可以使用更大的批次）

    print("train.py start")
    print("pretrained_model_path:",pretrained_model_path)
    print("data_dir:",data_dir)
    print("models_save_dir:",models_save_dir)

    # 调用train函数，传入预训练模型路径
    train()