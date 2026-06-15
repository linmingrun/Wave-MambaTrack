# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import math
import torch
import torch.nn as nn

from typing import List
from .utils import pos_to_pos_embed, logits_to_scores
from torch.utils.checkpoint import checkpoint

from .ffn import FFN
from .mlp import MLP
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid
from utils.box_ops import box_cxcywh_to_xyxy, box_iou_union

# ====================================================================================
# [创新点模块] Trajectory-Guided Query Attention (TGQA)
# 包含轨迹编码器和自适应门控融合机制
# ====================================================================================

class TrajectoryEncoder(nn.Module):
    def __init__(self, d_model=256, history_len=5):
        """
        将轨迹坐标序列编码成高维特征
        """
        super().__init__()
        self.history_len = history_len
        input_dim = history_len * 2  # (x, y) * len
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, track_history):
        # track_history: [Batch, Num_Tracks, history_len, 2]
        b, n, l, c = track_history.shape
        # 展平轨迹: [B, N, L*2]
        flat_traj = track_history.view(b, n, -1)
        # 编码
        return self.mlp(flat_traj)

class TrajectoryGuidedFusion(nn.Module):
    def __init__(self, d_model=256, history_len=5):
        super().__init__()
        self.traj_encoder = TrajectoryEncoder(d_model, history_len)
        
        # 自适应门控网络: 决定听视觉的还是听轨迹的
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid() # 输出 0~1 的权重
        )
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_query, track_history):
        """
        visual_query: [B, N, C]
        track_history: [B, N, L, 2]
        """
        # 1. 编码轨迹
        traj_embed = self.traj_encoder(track_history)
        
        # 2. 计算门控系数
        combined = torch.cat([visual_query, traj_embed], dim=-1)
        gate = self.gate_net(combined)
        
        # 3. 柔性融合 (基线保护机制：如果不准，Gate会自动趋向0)
        fused_query = visual_query + (gate * traj_embed)
        
        # 4. 投影与归一化
        output = self.norm(self.out_proj(fused_query))
        return output

# ====================================================================================

class QueryUpdater(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int,
                 tp_drop_ratio: float, fp_insert_ratio: float,
                 dropout: float,
                 use_checkpoint: bool, use_dab: bool,
                 update_threshold: float, long_memory_lambda: float,
                 visualize: bool = False):
        super(QueryUpdater, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.dropout = dropout

        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize

        self.update_threshold = update_threshold
        self.long_memory_lambda = long_memory_lambda

        # [新增] 初始化轨迹融合模块
        # history_len 必须与 RuntimeTracker 中设置的一致 (默认为 5)
        self.traj_fusion = TrajectoryGuidedFusion(d_model=self.hidden_dim, history_len=5)

        self.confidence_weight_net = nn.Sequential(
            MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, num_layers=2),
            nn.Sigmoid()
        )
        self.short_memory_fusion = MLP(input_dim=2*self.hidden_dim, hidden_dim=2*self.hidden_dim,
                                       output_dim=self.hidden_dim, num_layers=2)
        self.memory_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.memory_dropout = nn.Dropout(self.dropout)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_feat_dropout = nn.Dropout(self.dropout)
        self.query_feat_norm = nn.LayerNorm(self.hidden_dim)
        self.query_feat_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_pos_head = MLP(
            input_dim=self.hidden_dim*2,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )

        if self.use_dab is False:   # D-DETR, use this module to update the
            self.linear_pos1 = nn.Linear(256, 256)
            self.linear_pos2 = nn.Linear(256, 256)
            self.norm_pos = nn.LayerNorm(256)
            self.activation = nn.ReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                no_augment: bool = False):
        tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment)
        tracks = self.update_tracks_embedding(tracks=tracks)

        return tracks

    def update_tracks_embedding(self, tracks: List[TrackInstances]):
        for b in range(len(tracks)):
            scores = torch.max(logits_to_scores(logits=tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold
            if self.visualize:
                os.makedirs("./outputs/visualize_tmp/query_updater/", exist_ok=True)
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/current_ref_pts.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/current_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/current_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/current_labels.tensor")
                torch.save(tracks[b].scores.cpu(), "./outputs/visualize_tmp/query_updater/current_scores.tensor")
            
            if self.use_dab:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())
            else:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())

            query_pos = pos_to_pos_embed(tracks[b].ref_pts.sigmoid(), num_pos_feats=self.hidden_dim//2)
            output_embed = tracks[b].output_embed
            last_output_embed = tracks[b].last_output
            long_memory = tracks[b].long_memory.detach()

            # ===========================================================================
            # [创新点接入] Trajectory-Guided Query Enhancement
            # ===========================================================================
            # 检查当前 tracks 是否包含历史轨迹数据 (由 RuntimeTracker 维护)
            if hasattr(tracks[b], 'ref_pts_history'):
                # 获取历史轨迹: [Num_Tracks, history_len, 2]
                history = tracks[b].ref_pts_history
                
                # 调整维度以适配 Fusion 模块: [1, N, L, 2] -> 处理 -> [N, C]
                # 因为 update_tracks_embedding 是按 batch 循环的，这里相当于 batch=1
                enhanced_embed = self.traj_fusion(output_embed.unsqueeze(0), history.unsqueeze(0))
                
                # 更新 output_embed
                output_embed = enhanced_embed.squeeze(0)
            # ===========================================================================

            # Confidence Weight
            confidence_weight = self.confidence_weight_net(output_embed)

            # Adaptive Aggregation
            short_memory = self.short_memory_fusion(
                torch.cat((
                    confidence_weight * output_embed,
                    last_output_embed
                ), dim=-1)
            )

            # Query Feature Generate
            query_pos = self.query_pos_head(query_pos)
            q = short_memory + query_pos
            k = long_memory + query_pos
            tgt = output_embed
            # Attention
            tgt2 = self.memory_attn(q[None, :], k[None, :], tgt[None, :])[0][0, :]
            tgt = tgt + self.memory_dropout(tgt2)
            tgt = self.memory_norm(tgt)
            tgt = self.memory_ffn(tgt)
            # Long Memory ResNet
            query_feat = long_memory + self.query_feat_dropout(tgt)
            query_feat = self.query_feat_norm(query_feat)
            query_feat = self.query_feat_ffn(query_feat)

            # Update Long Memory
            long_memory = (1 - self.long_memory_lambda) * long_memory + \
                          self.long_memory_lambda * tracks[b].output_embed
            tracks[b].long_memory = tracks[b].long_memory * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    long_memory * is_pos.reshape((is_pos.shape[0], 1))
            # Update Last Outputs Embedding
            tracks[b].last_output = tracks[b].last_output * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    output_embed * is_pos.reshape((is_pos.shape[0], 1))

            if self.use_dab:
                tracks[b].query_embed[is_pos] = query_feat[is_pos]
            else:
                tracks[b].query_embed[:, self.hidden_dim:][is_pos] = query_feat[is_pos]
                # Update query pos, which is not appeared in DAB-D-DETR framework:
                new_query_pos = self.linear_pos2(self.activation(self.linear_pos1(output_embed)))
                query_pos = tracks[b].query_embed[:, :self.hidden_dim]
                query_pos = query_pos + new_query_pos
                query_pos = self.norm_pos(query_pos)
                tracks[b].query_embed[:, :self.hidden_dim][is_pos] = query_pos[is_pos]

            if self.visualize:
                torch.save(tracks[b].ref_pts.cpu(), "./outputs/visualize_tmp/query_updater/next_ref_pts.tensor")
                torch.save(tracks[b].query_embed.cpu(),
                           "./outputs/visualize_tmp/query_updater/next_query_feat.tensor")
                torch.save(tracks[b].ids.cpu(), "./outputs/visualize_tmp/query_updater/next_ids.tensor")
                torch.save(tracks[b].labels.cpu(), "./outputs/visualize_tmp/query_updater/next_labels.tensor")
                torch.save(scores.cpu(), "./outputs/visualize_tmp/query_updater/next_scores.tensor")

        return tracks

    def select_active_tracks(self, previous_tracks: List[TrackInstances],
                             new_tracks: List[TrackInstances],
                             unmatched_dets: List[TrackInstances],
                             no_augment: bool = False):
        tracks = []
        if self.training:
            for b in range(len(new_tracks)):
                # Update fields
                new_tracks[b].last_output = new_tracks[b].output_embed
                
                # [新增] 初始化新轨迹的历史记录
                # 训练时可能会有新轨迹加入，需要初始化它们的 history
                if not hasattr(new_tracks[b], 'ref_pts_history'):
                    device = new_tracks[b].output_embed.device
                    n = len(new_tracks[b])
                    # 用当前的中心点填充满历史，形状 [N, 5, 2]
                    current_pts = new_tracks[b].boxes[..., :2]
                    new_tracks[b].ref_pts_history = current_pts.unsqueeze(1).repeat(1, 5, 1)

                if self.use_dab:
                    new_tracks[b].long_memory = new_tracks[b].query_embed
                else:
                    new_tracks[b].long_memory = new_tracks[b].query_embed[:, self.hidden_dim:]
                unmatched_dets[b].last_output = unmatched_dets[b].output_embed
                
                # 这里的 unmatched 也需要 history，虽然它可能用不到
                if not hasattr(unmatched_dets[b], 'ref_pts_history'):
                    device = unmatched_dets[b].output_embed.device
                    n = len(unmatched_dets[b])
                    if n > 0:
                        current_pts = unmatched_dets[b].boxes[..., :2]
                        unmatched_dets[b].ref_pts_history = current_pts.unsqueeze(1).repeat(1, 5, 1)
                    else:
                        unmatched_dets[b].ref_pts_history = torch.zeros((0, 5, 2), device=device)

                if self.use_dab:
                    unmatched_dets[b].long_memory = unmatched_dets[b].query_embed
                else:
                    unmatched_dets[b].long_memory = unmatched_dets[b].query_embed[:, self.hidden_dim:]
                
                if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = TrackInstances.cat_tracked_instances(active_tracks, unmatched_dets[b])
                    scores = torch.max(logits_to_scores(logits=active_tracks.logits), dim=1).values
                    keep_idxes = (scores > self.update_threshold) | (active_tracks.ids >= 0)
                    active_tracks = active_tracks[keep_idxes]
                    active_tracks.ids[active_tracks.iou < 0.5] = -1
                else:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                    active_tracks = active_tracks[(active_tracks.iou > 0.5) & (active_tracks.ids >= 0)]
                    if self.tp_drop_ratio > 0.0 and not no_augment:
                        if len(active_tracks) > 0:
                            tp_keep_idx = torch.rand((len(active_tracks), )) > self.tp_drop_ratio
                            active_tracks = active_tracks[tp_keep_idx]
                    if self.fp_insert_ratio > 0.0 and not no_augment:
                        selected_active_tracks = active_tracks[
                            torch.bernoulli(
                                torch.ones((len(active_tracks), )) * self.fp_insert_ratio
                            ).bool()
                        ]
                        if len(unmatched_dets[b]) > 0 and len(selected_active_tracks) > 0:
                            fp_num = len(selected_active_tracks)
                            if fp_num >= len(unmatched_dets[b]):
                                insert_fp = unmatched_dets[b]
                            else:
                                selected_active_boxes = box_cxcywh_to_xyxy(selected_active_tracks.boxes)
                                unmatched_boxes = box_cxcywh_to_xyxy(unmatched_dets[b].boxes)
                                iou, _ = box_iou_union(unmatched_boxes, selected_active_boxes)
                                fp_idx = torch.max(iou, dim=0).indices
                                fp_idx = torch.unique(fp_idx)
                                insert_fp = unmatched_dets[b][fp_idx]
                            active_tracks = TrackInstances.cat_tracked_instances(active_tracks, insert_fp)

                if len(active_tracks) == 0:
                    device = next(self.query_feat_ffn.parameters()).device
                    fake_tracks = TrackInstances(frame_height=1.0, frame_width=1.0, hidden_dim=self.hidden_dim).to(
                        device=device)
                    if self.use_dab:
                        fake_tracks.query_embed = torch.randn((1, self.hidden_dim), dtype=torch.float,
                                                              device=device)
                    else:
                        fake_tracks.query_embed = torch.randn((1, 2 * self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.output_embed = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    if self.use_dab:
                        fake_tracks.ref_pts = torch.randn((1, 4), dtype=torch.float, device=device)
                    else:
                        # fake_tracks.ref_pts = torch.randn((1, 2), dtype=torch.float, device=device)
                        fake_tracks.ref_pts = torch.randn((1, 4), dtype=torch.float, device=device)
                    fake_tracks.ids = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.matched_idx = torch.as_tensor([-2], dtype=torch.long, device=device)
                    fake_tracks.boxes = torch.randn((1, 4), dtype=torch.float, device=device)
                    fake_tracks.logits = torch.randn((1, active_tracks.logits.shape[1]), dtype=torch.float, device=device)
                    fake_tracks.iou = torch.zeros((1,), dtype=torch.float, device=device)
                    fake_tracks.last_output = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    fake_tracks.long_memory = torch.randn((1, self.hidden_dim), dtype=torch.float, device=device)
                    # Fake track 也需要初始化 history
                    fake_tracks.ref_pts_history = torch.zeros((1, 5, 2), dtype=torch.float, device=device)
                    active_tracks = fake_tracks
                tracks.append(active_tracks)
        else:
            # Eval only has B=1.
            assert len(previous_tracks) == 1 and len(new_tracks) == 1
            new_tracks[0].last_output = new_tracks[0].output_embed
            
            # [新增] Eval 模式下初始化新轨迹的历史
            if not hasattr(new_tracks[0], 'ref_pts_history'):
                device = new_tracks[0].output_embed.device
                current_pts = new_tracks[0].boxes[..., :2]
                new_tracks[0].ref_pts_history = current_pts.unsqueeze(1).repeat(1, 5, 1)

            if self.use_dab:
                new_tracks[0].long_memory = new_tracks[0].query_embed
            else:
                new_tracks[0].long_memory = new_tracks[0].query_embed[:, self.hidden_dim:]
            active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[0], new_tracks[0])
            active_tracks = active_tracks[active_tracks.ids >= 0]
            tracks.append(active_tracks)
        return tracks


def build(config: dict):
    return QueryUpdater(
            hidden_dim=config["HIDDEN_DIM"],
            ffn_dim=config["FFN_DIM"],
            dropout=config["DROPOUT"],
            tp_drop_ratio=config["TP_DROP_RATE"] if "TP_DROP_RATE" in config else 0.0,
            fp_insert_ratio=config["FP_INSERT_RATE"] if "FP_INSERT_RATE" in config else 0.0,
            use_checkpoint=config["USE_CHECKPOINT"],
            use_dab=config["USE_DAB"],
            update_threshold=config["UPDATE_THRESH"],
            long_memory_lambda=config["LONG_MEMORY_LAMBDA"],
            visualize=config["VISUALIZE"]
        )