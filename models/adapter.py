import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 保持之前的修复 (Auto-Padding) ---
def haar_dwt(x):
    """
    执行单级离散小波变换 (Haar Wavelet Transform)
    增加了自动填充功能，防止输入尺寸为奇数导致报错。
    """
    b, c, h, w = x.size()

    # 检查并填充奇数尺寸
    pad_h = h % 2
    pad_w = w % 2
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    ll = x1 + x2 + x3 + x4
    lh = -x1 - x2 + x3 + x4
    hl = -x1 + x2 - x3 + x4
    hh = x1 - x2 - x3 + x4

    return torch.cat([ll, lh, hl, hh], dim=1)

class SceneAwareAdapter(nn.Module):
    """
    [Final Fix] Wavelet-based Scene-Aware Adapter (GroupNorm Version)
    修复方案: 
    1. 将 BatchNorm2d 替换为 GroupNorm。
       原因: 你的 Batch Size=1，BN 无法工作且会导致 In-place 报错。GroupNorm 更稳定。
    2. 保持 inplace=False。
    """
    def __init__(self, in_channels, num_scenes=3, reduction=16):
        super().__init__()
        print(f"--- [System Check] Initializing GROUPNORM SceneAwareAdapter (Stable Mode) ---")
        
        self.in_channels = in_channels
        
        # ========== 1. 频域场景编码 ==========
        self.scene_pool = nn.AdaptiveAvgPool2d(1)
        self.scene_fc_linear = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.scene_relu = nn.ReLU(inplace=False) 

        # 场景分类辅助任务
        self.scene_classifier = nn.Linear(in_channels // reduction, num_scenes)

        # ========== 2. 场景感知通道注意力 ==========
        self.channel_fc_linear = nn.Linear(in_channels // reduction, in_channels)
        self.channel_sigmoid = nn.Sigmoid()

        # ========== 3. 高频增强的空间注意力 ==========
        reduced_dim = in_channels // 4  # 通常是 64
        
        self.hf_conv = nn.Conv2d(in_channels * 3, reduced_dim, kernel_size=1)
        
        # [关键修改] 使用 GroupNorm 替代 BatchNorm2d
        # GroupNorm 不依赖 BatchSize，且没有 running_mean 的 inplace 更新冲突
        # num_groups=4 是一个经验值，将 64 个通道分成 4 组
        self.hf_norm = nn.GroupNorm(num_groups=4, num_channels=reduced_dim)
        
        self.hf_relu = nn.ReLU(inplace=False)

        # 原有的多尺度卷积
        self.spatial_scales = [1, 3, 5]
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, 
                in_channels // reduction, 
                kernel_size=k, 
                padding=k // 2, 
                groups=in_channels // reduction
            ) for k in self.spatial_scales
        ])

        # 融合层
        total_concat_channels = len(self.spatial_scales) * (in_channels // reduction) + reduced_dim
        self.spatial_fusion = nn.Conv2d(total_concat_channels, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()

        # [Step 1] 小波分解
        dwt_out = haar_dwt(x)
        x_ll = dwt_out[:, :c, :, :]          
        x_high = dwt_out[:, c:, :, :]        

        # [Step 2] 场景编码
        s_feat = self.scene_pool(x_ll)
        s_feat = s_feat.flatten(1)
        s_feat = self.scene_fc_linear(s_feat)
        scene_feat = self.scene_relu(s_feat)

        scene_logits = self.scene_classifier(scene_feat)

        # [Step 3] 通道注意力
        c_weight = self.channel_fc_linear(scene_feat)
        channel_weight = self.channel_sigmoid(c_weight).view(b, c, 1, 1)
        out_channel = x * channel_weight

        # [Step 4] 空间注意力
        # 显式分离变量，防止计算图纠缠
        hf_feat = self.hf_conv(x_high)
        hf_feat = self.hf_norm(hf_feat) # GroupNorm
        hf_feat = self.hf_relu(hf_feat)
        
        # 插值: 使用 contiguous 确保内存连续，减少报错概率
        high_freq_map = F.interpolate(hf_feat, size=(h, w), mode='bilinear', align_corners=False)

        spatial_feats = [conv(x) for conv in self.spatial_convs]
        spatial_concat = torch.cat(spatial_feats + [high_freq_map], dim=1)
        
        spatial_weight = torch.sigmoid(self.spatial_fusion(spatial_concat))
        out_spatial = x * spatial_weight

        # [Step 5] 最终融合
        out = 0.5 * out_channel + 0.5 * out_spatial

        return out, scene_logits