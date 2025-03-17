import torch
import torch.nn as nn
import torch.nn.functional as F

class TextureAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        # 使用不同尺度的空洞卷积捕获多尺度纹理特征
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=r, dilation=r)
            for r in [1, 2, 4, 8]
        ])
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        feat = self.conv1(x)
        # 多尺度特征提取
        texture_feats = []
        for conv in self.dilated_convs:
            texture_feats.append(conv(feat))
        # 特征融合
        texture_feat = torch.cat(texture_feats, dim=1)
        attention = torch.sigmoid(self.conv2(texture_feat))
        return x + self.gamma * (x * attention)

class TextureEnhancedDenseNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        # 获取DenseNet的特征提取层
        self.features = base_model.features
        
        # 获取DenseNet各个阶段的输出通道数
        self.transition1_out = 128  # 第一个transition layer后的通道数
        self.transition2_out = 256  # 第二个transition layer后的通道数
        self.transition3_out = 512  # 第三个transition layer后的通道数
        self.final_out = 1024      # 最后一个dense block后的通道数
        
        # 在dense block之后添加纹理注意力模块
        self.texture_attention1 = TextureAttention(self.transition1_out)
        self.texture_attention2 = TextureAttention(self.transition2_out)
        self.texture_attention3 = TextureAttention(self.transition3_out)
        self.texture_attention4 = TextureAttention(self.final_out)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(self.final_out, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 前向传播通过dense blocks和纹理注意力模块
        features = self.features[:6](x)        # 第一个transition layer
        features = self.texture_attention1(features)
        
        features = self.features[6:8](features) # 第二个transition layer
        features = self.texture_attention2(features)
        
        features = self.features[8:10](features) # 第三个transition layer
        features = self.texture_attention3(features)
        
        features = self.features[10:](features)  # 最后一个dense block
        features = self.texture_attention4(features)
        
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


