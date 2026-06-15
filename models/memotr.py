# @Author       : Ruopeng Gao
# @Date         : 2022/9/4
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# [修复] 确保导入 List
from typing import List

from .mlp import MLP
from .ffn import FFN
from .backbone import BackboneWithPE
from .deformable_transformer import DeformableTransformer
from .query_updater import build as build_query_updater
from .utils import get_clones, pos_to_pos_embed

from .backbone import build as build_backbone_with_pe
from .deformable_transformer import build as build_deformable_transformer

# [新增] 导入你的创新模块
# 确保你的 adapter.py 和 bi_mamba.py 已经在 models/ 目录下
from .adapter import SceneAwareAdapter
from .temporal_mamba import TemporalAwareMamba

from utils.nested_tensor import NestedTensor
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid

from torch.utils.checkpoint import checkpoint


class MeMOTR(nn.Module):
    def __init__(self, backbone: BackboneWithPE, transformer: DeformableTransformer,
                 query_updater: nn.Module,
                 num_classes: int, n_det_queries: int, n_feature_levels: int,
                 hidden_dim: int, ffn_dim: int, dropout: float,
                 aux_loss: bool = True, with_box_refine: bool = True,
                 use_checkpoint: bool = False, checkpoint_level: int = 2,
                 use_dab: bool = False,
                 visualize: bool = False):
        super(MeMOTR, self).__init__()

        self.num_classes = num_classes
        self.n_det_queries = n_det_queries
        self.n_feature_levels = n_feature_levels
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.use_checkpoint = use_checkpoint
        self.checkpoint_level = checkpoint_level
        self.use_dab = use_dab
        self.visualize = visualize

        # Net:
        self.backbone = backbone
        self.transformer = transformer
        self.query_updater = query_updater
        
        # =====================================================================
        # [创新点 1] 初始化 Scene Adapter
        # 放置在 Feature Projection 之后，所以输入通道是 hidden_dim (256)
        # =====================================================================
        self.scene_adapter = SceneAwareAdapter(in_channels=hidden_dim, num_scenes=3)

        # =====================================================================
        # [创新点 2] 初始化Mamba
        # 用于增强 Transformer 输出的 Query 特征
        # =====================================================================
        self.temporal_mamba = TemporalAwareMamba(
            d_model=hidden_dim, 
            d_state=32, 
            n_det_queries=self.n_det_queries
        )

        self.class_embed = nn.Linear(in_features=self.hidden_dim, out_features=num_classes)
        self.bbox_embed = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
        if self.use_dab:
            self.det_anchor = nn.Parameter(torch.randn(self.n_det_queries, 4))  # (N_det, 4)
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim))       # (N_det, C)
        else:
            self.det_query_embed = nn.Parameter(torch.randn(self.n_det_queries, self.hidden_dim * 2))   # (N_det, 2C)
        assert self.n_feature_levels > 1
        n_backbone_inter_layers = backbone.n_inter_layers()
        n_backbone_inter_channels = backbone.n_inter_channels()
        feature_proj_list = []
        for i in range(n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[i], out_channels=self.hidden_dim, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        for _ in range(self.n_feature_levels - n_backbone_inter_layers):
            feature_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels=n_backbone_inter_channels[-1], out_channels=self.hidden_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim)
            ))
        self.feature_projs = nn.ModuleList(feature_proj_list)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.feature_projs:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        if self.with_box_refine:
            self.class_embed = get_clones(self.class_embed, self.transformer.get_n_dec_layers())
            self.bbox_embed = get_clones(self.bbox_embed, self.transformer.get_n_dec_layers())
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.set_refine_bbox_embed(self.bbox_embed)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(self.transformer.get_n_dec_layers())])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(self.transformer.get_n_dec_layers())])

    # [修复] 将 tracks: list[TrackInstances] 改为 List[TrackInstances] 以兼容旧版Python
    def forward(self, frame: NestedTensor, tracks: List[TrackInstances]):
        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/memotr/", exist_ok=True)

        # 图像经过 backbone
        if self.use_checkpoint and self.checkpoint_level != 3:
            features, pos = checkpoint(self.backbone, frame, use_reentrant=False)
        else:
            features, pos = self.backbone(frame)

        srcs, masks = [], []
        # 用于存储 Scene Adapter 预测的场景分类结果 (用于辅助 Loss)
        self.current_scene_logits = None

        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            # 1. 先投影到 unified hidden_dim (256)
            src_proj = self.feature_projs[layer](src)
            
            # =================================================================
            # [创新点 1 接入] Scene Adapter
            # 对特征进行“自适应清洗”，去除水下浑浊/光照影响
            # =================================================================
            src_adapted, scene_logits = self.scene_adapter(src_proj)
            
            # 如果是最后一层特征，记录下 Logits (假设场景主要由深层语义决定)
            if layer == len(features) - 1:
                self.current_scene_logits = scene_logits

            # 将清洗后的特征送入列表
            srcs.append(src_adapted)
            masks.append(mask)

        if self.n_feature_levels > len(srcs):
            srcs_len = len(srcs)
            for layer in range(srcs_len, self.n_feature_levels):
                if layer == srcs_len:
                    # 注意：这里需要先把 features[-1] 投影，再通过 Adapter
                    src_raw = features[-1].tensors
                    src_proj = self.feature_projs[layer](src_raw)
                else:
                    # 对于更高层的 FPN，输入是上一层的 src (已经是 adapted 过的了)
                    # 为了简单，我们对 FPN 生成的层再过一次 Adapter，或者复用上一层的适配策略
                    # 这里选择：Projection -> Adapter
                    src_prev = srcs[-1]
                    src_proj = self.feature_projs[layer](src_prev)

                # [创新点 1 接入] 即使是 FPN 生成的层，也过一遍 Adapter 增强
                src_adapted, _ = self.scene_adapter(src_proj)

                mask = frame.masks
                mask = F.interpolate(mask[None, ...].float(), size=src_adapted.shape[-2:])[0].to(torch.bool)
                pos.append(self.backbone.position_embedding(NestedTensor(src_adapted, mask)).to(src_adapted.device))
                srcs.append(src_adapted)
                masks.append(mask)
        
        # srcs is n_feature_levels * [(B, C, H, W)]
        # masks is n_feature_levels * [(B, H, W)]
        # pos is n_features_levels * [(B, C, H, W)]

        reference_points = self.get_reference_points(tracks=tracks).to(srcs[0].device)      # (B, Nd+Nq, 2/4)
        query_embed = self.get_query_embed(tracks=tracks).to(srcs[0].device)
        query_mask = self.get_query_mask(tracks=tracks).to(srcs[0].device)                  # (B, Nd+Nq)

        # DETR:
        outputs, init_reference, inter_references, inter_queries = self.transformer(
            srcs=srcs,
            masks=masks,
            pos_embeds=pos,
            query_embed=query_embed,
            ref_pts=reference_points,
            query_mask=query_mask
        )
        # outputs: (n_dec_layers, B, Nd+Nq, C)
        
        # ==================== 【修正后的代码】 ====================
        # 1. 取出最后一层的输出
        final_output = outputs[-1]  # [B, N, C]
        
        # 2. Mamba 增强
        enhanced_output = self.temporal_mamba(final_output)
        
        # 3. 【关键修复】使用 torch.cat 创建一个新的 Tensor，而不是修改旧的
        # outputs 的形状是 [Layers, B, N, C]
        # 我们把 前面几层 (outputs[:-1]) 和 增强后的最后一层 拼接起来
        
        # 先把 enhanced_output 变回 [1, B, N, C] 以便拼接
        enhanced_output_unsqueezed = enhanced_output.unsqueeze(0)
        
        # 拼接：前 n-1 层 + 新的第 n 层
        outputs = torch.cat([outputs[:-1], enhanced_output_unsqueezed], dim=0)
        # ========================================================

        # init_reference: (B, Nd+Nq, 2)
        # inter_references: (n_dec_layers, B, Nd+Nq, 4)
        output_classes, output_bboxes = [], []
        assert outputs.ndim == 4, f"Deformable Transformer's outputs should have shape (n_dec_layers, B, Nd+Nq, C, " \
                                  f"but get n_dim={outputs.ndim}"
        for level in range(outputs.shape[0]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[level - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[level](outputs[level])
            bbox_tmp = self.bbox_embed[level](outputs[level])
            if reference.shape[-1] == 4:
                bbox_tmp += reference
            else:
                assert reference.shape[-1] == 2, f"Reference should have only 2 coord, but get {reference.shape[-1]}."
                bbox_tmp[..., :2] += reference
            output_bbox = bbox_tmp.sigmoid()
            output_classes.append(output_class)
            output_bboxes.append(output_bbox)

            if self.visualize:
                torch.save(reference[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/detection_ref_pts_layer_{level}.tensor")
                torch.save(reference[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/track_ref_pts_layer_{level}.tensor")
                torch.save(output_class[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/detection_logits_layer_{level}.tensor")
                torch.save(output_class[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/track_logits_layer_{level}.tensor")
                torch.save(output_bbox[0, :self.n_det_queries, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/detection_boxes_layer_{level}.tensor")
                torch.save(output_bbox[0, self.n_det_queries:, :].cpu(),
                           f"./outputs/visualize_tmp/memotr/track_boxes_layer_{level}.tensor")

        output_classes = torch.stack(output_classes, dim=0)
        output_bboxes = torch.stack(output_bboxes, dim=0)
        res = {
            "pred_logits": output_classes[-1],
            "pred_bboxes": output_bboxes[-1],
            "last_ref_pts": inverse_sigmoid(inter_references[-2, :, :, :]) if self.use_dab       # (B, Nd+Nq, 4)
            else inverse_sigmoid(inter_references[-2, :, :, :]),                                 # (B, Nd+Nq, 2)
            "query_mask": query_mask,                   # (B, Nd+Nq)
            "det_query_embed": query_embed[0][:self.n_det_queries],
            "init_ref_pts": inverse_sigmoid(init_reference)
        }
        
        # [新增] 将 Scene Adapter 的预测结果放入输出字典 (供 Loss 计算使用)
        if self.current_scene_logits is not None:
            res['pred_scene_logits'] = self.current_scene_logits

        if self.aux_loss:
            res["aux_outputs"] = self.set_aux_loss(output_classes=output_classes,
                                                   output_bboxes=output_bboxes,
                                                   query_mask=query_mask,
                                                   queries=inter_queries)
        res["outputs"] = outputs[-1]     # (B, Nd+Nq, C)
        return res

    @torch.jit.unused
    def set_aux_loss(self, output_classes, output_bboxes, query_mask, queries):
        """
        this is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [
            {"pred_logits": a, "pred_bboxes": b, "query_mask": query_mask, "queries": c}
            for a, b, c in zip(output_classes[:-1], output_bboxes[:-1], queries[1:])
        ]

    def get_det_reference_points(self) -> torch.Tensor:
        """
        Returns: (Nd, 2)
        """
        if self.use_dab:
            return self.det_anchor
        else:
            return self.transformer.reference_points(self.det_query_embed[:, :self.hidden_dim])

    def get_track_reference_points(self, tracks: List[TrackInstances]):
        """
        Returns: (B, Nq, 2/4)
        """
        max_len = max([len(t.ref_pts) for t in tracks])
        if self.use_dab:
            references = torch.zeros((len(tracks), max_len, 4))
        else:
            # references = torch.zeros((len(tracks), max_len, 2))
            references = torch.zeros((len(tracks), max_len, 4))
        for i in range(len(tracks)):
            references[i, :len(tracks[i].ref_pts), :] = tracks[i].ref_pts
        return references

    def get_track_query_embed(self, tracks: List[TrackInstances]):
        """
        Returns: (B, Nq, 2C)
        """
        max_len = max([len(t.query_embed) for t in tracks])
        if self.use_dab:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim))
        else:
            query_embed = torch.zeros((len(tracks), max_len, self.hidden_dim * 2))
        for i in range(len(tracks)):
            query_embed[i, :len(tracks[i].query_embed), :] = tracks[i].query_embed
        return query_embed

    def get_reference_points(self, tracks: List[TrackInstances]):
        det_references = self.get_det_reference_points().repeat(len(tracks), 1, 1)                      # (B, Nd, 2)
        if det_references.shape[-1] == 2:
            det_references = torch.cat(
                (det_references, torch.zeros_like(det_references, device=det_references.device)),
                dim=-1
            )
        track_references = self.get_track_reference_points(tracks=tracks).to(det_references.device)     # (B, Nq, 2)
        return torch.cat((det_references, track_references), dim=1)

    def get_query_embed(self, tracks: List[TrackInstances]):
        """
        Returns: (B, Nd+Nq, 2C)
        """
        if self.use_dab:
            det_query_embed = self.det_query_embed
            det_query_embed = det_query_embed.repeat(len(tracks), 1, 1)
        else:
            det_query_embed = self.det_query_embed.repeat(len(tracks), 1, 1)                    # (B, Nd, 2C)
        track_query_embed = self.get_track_query_embed(tracks).to(det_query_embed.device)       # (B, Nq, 2C)
        return torch.cat((det_query_embed, track_query_embed), dim=1)

    def get_query_mask(self, tracks: List[TrackInstances]):
        """
        Returns: (B, Nd+Nq)
        """
        track_max_len = max([len(t.query_embed) for t in tracks])
        det_query_mask = torch.zeros((len(tracks), self.n_det_queries)).to(torch.bool)
        track_query_mask = torch.zeros((len(tracks), track_max_len))
        for i in range(len(tracks)):
            if len(tracks[i].query_embed) > 0:
                track_query_mask[i, len(tracks[i].query_embed):] = 1
        track_query_mask = track_query_mask.to(torch.bool)
        return torch.cat((det_query_mask, track_query_mask), dim=1).to(self.det_query_embed.device)

    def postprocess_single_frame(self, previous_tracks: List[TrackInstances],
                                 new_tracks: List[TrackInstances],
                                 unmatched_dets: List[TrackInstances] | None,
                                 no_augment: bool = False):
        """
        Query updating.
        """
        return self.query_updater(previous_tracks, new_tracks, unmatched_dets, no_augment)


def build(config: dict):
    dataset_num_classes = {
        "DanceTrack": 1,
        "SportsMOT": 1,
        "MOT17": 1,
        "MOT17_SPLIT": 1,
        "BDD100K": 8,
    }
    assert config["DATASET"] in dataset_num_classes, f"Do not know the class num of {config['DATASET']} dataset."
    num_classes = dataset_num_classes[config["DATASET"]]

    backbone_with_pe = build_backbone_with_pe(config=config)
    deformable_transformer = build_deformable_transformer(config=config)
    query_updater = build_query_updater(config=config)

    return MeMOTR(
        backbone=backbone_with_pe,
        transformer=deformable_transformer,
        query_updater=query_updater,
        num_classes=num_classes,
        n_det_queries=config["NUM_DET_QUERIES"],
        n_feature_levels=config["NUM_FEATURE_LEVELS"],
        hidden_dim=config["HIDDEN_DIM"],
        ffn_dim=config["FFN_DIM"],
        dropout=config["DROPOUT"],
        aux_loss=True,
        with_box_refine=True,
        use_checkpoint=config["USE_CHECKPOINT"],
        checkpoint_level=config["CHECKPOINT_LEVEL"],
        use_dab=config["USE_DAB"],
        visualize=config["VISUALIZE"]
    )