import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoNetModel(nn.Module):
    def __init__(self, num_classes=10, input_size=(16, 224, 224)):
        super(VideoNetModel, self).__init__()
        
        # 轻量级视频分类网络设计，适合边缘端部署
        # 使用时空分离卷积减少计算量
        self.temporal_kernel_size = 3
        self.spatial_kernel_size = 3
        
        # 第一层：混合卷积层 - 先用3D卷积捕获初步时空特征
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),  # 空间卷积为主
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),  # 时间卷积
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        )
        
        # 设计轻量级Residual Block，使用分离卷积
        class LightweightResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=(1,1,1), downsample=None):
                super(LightweightResidualBlock, self).__init__()
                
                # 分离卷积设计：先时间卷积，再空间卷积
                self.conv1 = nn.Sequential(
                    # 时间卷积
                    nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), 
                              stride=(stride[0], 1, 1), padding=(1, 0, 0), groups=in_channels),
                    nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                    # 空间卷积
                    nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), 
                              stride=(1, stride[1], stride[2]), padding=(0, 1, 1), groups=out_channels),
                    nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                    nn.BatchNorm3d(out_channels)
                )
                
                self.downsample = downsample
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                identity = x
                out = self.conv1(x)
                
                if self.downsample is not None:
                    identity = self.downsample(identity)
                
                out += identity
                out = self.relu(out)
                return out
        
        # 轻量级时间注意力模块
        class LightweightTemporalAttention(nn.Module):
            def __init__(self, in_channels):
                super(LightweightTemporalAttention, self).__init__()
                # 使用全局平均池化简化计算
                self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # 保留时间维度
                self.conv = nn.Conv3d(in_channels, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                # 计算时间维度上的注意力权重
                attention = self.pool(x)  # (B, C, T, 1, 1)
                attention = self.sigmoid(self.conv(attention))
                # 应用注意力权重
                return x * attention
        
        # 构建网络层
        self.layer1 = self._make_layer(32, 64, LightweightResidualBlock, 2, stride=(1,2,2))
        self.temporal_attention1 = LightweightTemporalAttention(64)
        
        self.layer2 = self._make_layer(64, 128, LightweightResidualBlock, 2, stride=(2,2,2))
        self.layer3 = self._make_layer(128, 256, LightweightResidualBlock, 2, stride=(2,2,2))
        
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 动态计算特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1], input_size[2])
            out = self._forward_features(dummy_input)
            self.feature_size = out.view(1, -1).size(1)
        
        print(f"Feature size: {self.feature_size}")
        
        # 轻量级全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 减少dropout比例以提高性能
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, block, blocks, stride=(1,1,1)):
        downsample = None
        if stride != (1,1,1) or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.temporal_attention1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        return x
    
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x