import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba.mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("【警告】未安装 mamba_ssm，请执行: pip install mamba-ssm")


class MQIMBlock(nn.Module):
    """
    Mamba Query Interaction Module
    【修复】移除所有可能导致 inplace 操作的代码
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("必须安装 mamba_ssm!")
        
        self.norm = nn.LayerNorm(d_model)
        self.ssm_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ssm_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.z_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q):
        """
        Args:
            q: [B, N, D]
        Returns:
            [B, N, D]
        """
        # ========== 修复 1: 避免 clone，直接使用 norm 输出 ==========
        q_norm = self.norm(q)
        
        # ========== 修复 2: flip 后立即 contiguous，避免 stride 问题 ==========
        x_fwd = self.ssm_fwd(q_norm)
        
        # 反向处理：flip -> contiguous -> ssm -> flip -> contiguous
        q_flipped = q_norm.flip(dims=[1]).contiguous()
        x_bwd_flipped = self.ssm_bwd(q_flipped)
        x_bwd = x_bwd_flipped.flip(dims=[1]).contiguous()
        
        x_ssm = x_fwd + x_bwd
        
        # 门控
        z = F.silu(self.z_proj(q_norm))
        combined = x_ssm * z
        
        # ========== 修复 3: 使用显式加法，不要 += ==========
        out = self.out_proj(combined)
        return q + out  # 不要写成 q += out


class TemporalAwareMamba(nn.Module):
    """
    【修复版】避免所有 inplace 操作
    """
    def __init__(self, d_model=256, d_state=16, n_det_queries=300):
        super().__init__()
        self.n_det = n_det_queries
        
        self.temporal_mqim = MQIMBlock(d_model, d_state)
        self.spatial_mqim = MQIMBlock(d_model, d_state)
        self.fusion = nn.Linear(d_model * 2, d_model)
    
    def forward(self, queries):
        """
        Args:
            queries: [B, N, D]
        Returns:
            [B, N, D]
        """
        B, N, D = queries.shape
        
        # ===== 空间交互 =====
        spatial_out = self.spatial_mqim(queries)
        
        # ===== 时序交互（仅 Track Query）=====
        if N > self.n_det:
            # ========== 关键修复：不要原地修改 spatial_out ==========
            # 错误写法：spatial_out[:, self.n_det:, :] = enhanced_track
            
            # 正确写法：分离、处理、重新拼接
            det_part = spatial_out[:, :self.n_det, :]  # 检测部分
            track_queries = queries[:, self.n_det:, :]  # Track Query 原始输入
            
            # Track 时序增强
            temporal_track = self.temporal_mqim(track_queries)
            
            # 融合空间和时序信息
            spatial_track = spatial_out[:, self.n_det:, :]
            fused = torch.cat([spatial_track, temporal_track], dim=-1)
            enhanced_track = self.fusion(fused)
            
            # ========== 修复：使用 cat 而不是原地赋值 ==========
            final_out = torch.cat([det_part, enhanced_track], dim=1)
            return final_out
        
        return spatial_out