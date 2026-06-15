import torch
import torch.nn as nn

class TrajectoryEncoder(nn.Module):
    def __init__(self, d_model=256, history_len=10):
        """
        将轨迹坐标序列编码成高维特征
        history_len: 回溯过去多少帧 (建议 5-10)
        """
        super().__init__()
        self.history_len = history_len
        
        # 输入维度: history_len * 2 (x, y)
        input_dim = history_len * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model) # 再次Norm，保证数值稳定
        )

    def forward(self, track_history):
        # track_history shape: [Batch, Num_Tracks, history_len, 2]
        b, n, l, c = track_history.shape
        
        # 展平: [B, N, L*2]
        flat_traj = track_history.view(b, n, -1)
        
        # 编码: [B, N, d_model]
        traj_embed = self.mlp(flat_traj)
        return traj_embed


class TrajectoryGuidedFusion(nn.Module):
    def __init__(self, d_model=256, history_len=10):
        super().__init__()
        
        # 1. 轨迹编码器
        self.traj_encoder = TrajectoryEncoder(d_model, history_len)
        
        # 2. 自适应门控网络 (Adaptive Gating Network)
        # 输入: 视觉特征 + 轨迹特征
        # 输出: 门控系数 (0~1)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model), # 这是一个 Channel-wise 的门控
            nn.Sigmoid() # 关键！把值压到 0-1 之间
        )
        
        # 3. 融合后的投影层 (可选，为了平滑)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_query, track_history):
        """
        visual_query: 来自 Transformer 的视觉特征 [B, N, C]
        track_history: 过去几帧的中心点坐标 [B, N, L, 2]
        """
        # 1. 获取轨迹特征
        traj_embed = self.traj_encoder(track_history)
        
        # 2. 计算门控系数 Gate
        # 拼接视觉和轨迹，让网络自己判断谁更重要
        combined = torch.cat([visual_query, traj_embed], dim=-1)
        gate = self.gate_net(combined) # [B, N, C]
        
        # 3. 融合 (Soft Fusion)
        # 逻辑：原特征 + (Gate * 轨迹特征)
        # 这样即使 Gate 全为 0，也至少保留了原来的视觉特征，不会导致性能下降
        fused_query = visual_query + (gate * traj_embed)
        
        # 4. 残差连接 + Norm
        output = self.norm(self.out_proj(fused_query))
        
        return output