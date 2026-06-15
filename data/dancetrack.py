# @Author       : Ruopeng Gao
# @Date         : 2022/8/30
import os
from math import floor
from random import randint

import torch
from PIL import Image
import data.transforms as T
from .mot import MOTDataset
from collections import defaultdict
from typing import List  # 必须导入 List
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


class DanceTrack(MOTDataset):
    def __init__(self, config: dict, split: str, transform):
        super(DanceTrack, self).__init__(config=config, split=split, transform=transform)

        self.config = config
        self.transform = transform
        self.dataset_name = config["DATASET"]
        assert split == "train" or split == "test", f"Split {split} is not supported!"
        self.split_dir = os.path.join(config["DATA_ROOT"], self.dataset_name, split)
        assert os.path.exists(self.split_dir), f"Dir {self.split_dir} is not exist."

        # Sampling setting.
        self.sample_steps: list = config["SAMPLE_STEPS"]
        self.sample_intervals: list = config["SAMPLE_INTERVALS"]
        self.sample_modes: list = config["SAMPLE_MODES"]
        self.sample_lengths: list = config["SAMPLE_LENGTHS"]
        self.sample_stage = None
        self.sample_begin_frames = None
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        self.sample_vid_tmax = None

        self.gts = defaultdict(lambda: defaultdict(list))
        self.vid_idx = dict()
        self.idx_vid = dict()

        for vid in os.listdir(self.split_dir):
            gt_path = os.path.join(self.split_dir, vid, "gt", "gt.txt")
            
            # --- [修复核心] 兼容性更强的数据读取逻辑 ---
            if not os.path.exists(gt_path):
                print(f"Warning: GT file not found: {gt_path}")
                continue

            for line in open(gt_path):
                line = line.strip()
                if not line: continue

                # 1. 自动判断分隔符 (空格或逗号)
                if "," in line:
                    parts = line.split(",")
                else:
                    parts = line.split()
                
                # 2. 确保至少有6列数据 (frame, id, x, y, w, h)
                if len(parts) < 6:
                    continue

                try:
                    # 3. 解析基础数据 (使用 float 转换坐标，防止 '12.0' 报错)
                    t = int(float(parts[0]))  # 先转float再转int，兼容 '1.0'
                    i = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])

                    # 4. 修正帧号：如果从0开始，强制+1以适应MeMOTR
                    if t == 0:
                        t = 1
                    
                    # 5. 存入字典
                    self.gts[vid][t].append([i, x, y, w, h])
                except ValueError as e:
                    print(f"Warning: Skipping invalid line: {line} | Error: {e}")
                    continue
            # --- [修复结束] ---

        vids = list(self.gts.keys())

        for vid in vids:
            self.vid_idx[vid] = len(self.vid_idx)
            self.idx_vid[self.vid_idx[vid]] = vid

        self.set_epoch(0)

        return

    def __getitem__(self, item):
        vid, begin_frame = self.sample_begin_frames[item]
        frame_idxs = self.sample_frames_idx(vid=vid, begin_frame=begin_frame)
        imgs, infos = self.get_multi_frames(vid=vid, idxs=frame_idxs)
        if self.transform is not None:
            imgs, infos = self.transform(imgs, infos)
        return {
            "imgs": imgs,
            "infos": infos
        }

    def __len__(self):
        assert self.sample_begin_frames is not None, "Please use set_epoch to init DanceTrack Dataset."
        return len(self.sample_begin_frames)

    # [修复] 这里 list[int] 改为了 List[int]
    def sample_frames_idx(self, vid: int, begin_frame: int) -> List[int]:
        if self.sample_mode == "random_interval":
            assert self.sample_length > 1, "Sample length is less than 2."
            remain_frames = self.sample_vid_tmax[vid] - begin_frame
            max_interval = floor(remain_frames / (self.sample_length - 1))
            interval = min(randint(1, self.sample_interval), max_interval)
            frame_idxs = [begin_frame + interval * i for i in range(self.sample_length)]
            return frame_idxs
        else:
            raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

    def set_epoch(self, epoch: int):
        self.sample_begin_frames = list()
        self.sample_vid_tmax = dict()
        self.sample_stage = 0
        for step in self.sample_steps:
            if epoch >= step:
                self.sample_stage += 1
        assert self.sample_stage < len(self.sample_steps) + 1
        self.sample_length = self.sample_lengths[min(len(self.sample_lengths) - 1, self.sample_stage)]
        self.sample_mode = self.sample_modes[min(len(self.sample_modes) - 1, self.sample_stage)]
        self.sample_interval = self.sample_intervals[min(len(self.sample_intervals) - 1, self.sample_stage)]
        for vid in self.vid_idx.keys():
            # [增加保护] 防止空视频导致 crash
            if not self.gts[vid]:
                print(f"Warning: Video {vid} has no GT data!")
                continue
                
            t_min = min(self.gts[vid].keys())
            t_max = max(self.gts[vid].keys())
            self.sample_vid_tmax[vid] = t_max
            for t in range(t_min, t_max - (self.sample_length - 1) + 1):
                self.sample_begin_frames.append((vid, t))

        return

    def get_single_frame(self, vid: str, idx: int):
        # 优先尝试 8位 格式
        img_path_8 = os.path.join(self.split_dir, vid, "img1", f"{idx:08d}.jpg")
        # 其次尝试 6位 格式
        img_path_6 = os.path.join(self.split_dir, vid, "img1", f"{idx:06d}.jpg")
        # 最后尝试 无补零 格式
        img_path_raw = os.path.join(self.split_dir, vid, "img1", f"{idx}.jpg")
        
        if os.path.exists(img_path_8):
            img_path = img_path_8
        elif os.path.exists(img_path_6):
            img_path = img_path_6
        elif os.path.exists(img_path_raw):
            img_path = img_path_raw
        else:
            # 默认回退，或者报错提示
            img_path = img_path_8 
            # print(f"Warning: Image not found {img_path}")

        img = Image.open(img_path)
        info = {}
        ids_offset = self.vid_idx[vid] * 100000

        # 真值：
        info["boxes"] = list()
        info["ids"] = list()
        info["labels"] = list()
        info["areas"] = list()
        info["frame_idx"] = torch.as_tensor(idx)

        for i, *xywh in self.gts[vid][idx]:
            info["boxes"].append(list(map(float, xywh)))
            info["areas"].append(xywh[2] * xywh[3])     # area = w * h
            info["ids"].append(i + ids_offset)
            info["labels"].append(0)                    # DanceTrack, all people.
        info["boxes"] = torch.as_tensor(info["boxes"])
        info["areas"] = torch.as_tensor(info["areas"])
        info["ids"] = torch.as_tensor(info["ids"])
        info["labels"] = torch.as_tensor(info["labels"])
        # xywh to x1y1x2y2
        if len(info["boxes"]) > 0:
            info["boxes"][:, 2:] += info["boxes"][:, :2]
        else:
            info["boxes"] = torch.zeros((0, 4))
            info["ids"] = torch.zeros((0,), dtype=torch.long)
            info["labels"] = torch.zeros((0,), dtype=torch.long)

        return img, info

    # [修复] 这里 list[int] 改为了 List[int]
    def get_multi_frames(self, vid: str, idxs: List[int]):
        return zip(*[self.get_single_frame(vid=vid, idx=i) for i in idxs])


def transfroms_for_train(coco_size: bool = False, overflow_bbox: bool = False, reverse_clip: bool = False):
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]  # from MOTR
    return T.MultiCompose([
        T.MultiRandomHorizontalFlip(),
        T.MultiRandomSelect(
            T.MultiRandomResize(sizes=scales, max_size=1536),
            T.MultiCompose([
                T.MultiRandomResize([400, 500, 600] if coco_size else [800, 1000, 1200]),
                T.MultiRandomCrop(
                    min_size=384 if coco_size else 800,
                    max_size=600 if coco_size else 1200,
                    overflow_bbox=overflow_bbox
                ),
                T.MultiRandomResize(sizes=scales, max_size=1536)
            ])
        ),
        T.MultiHSV(),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        T.MultiReverseClip(reverse=reverse_clip)
    ])


def transforms_for_eval():
    return T.MultiCompose([
        T.MultiRandomResize(sizes=[800], max_size=1333),
        T.MultiCompose([
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ])


def build(config: dict, split: str):
    if split == "train":
        return DanceTrack(
            config=config,
            split=split,
            transform=transfroms_for_train(
                coco_size=config["COCO_SIZE"],
                overflow_bbox=config["OVERFLOW_BBOX"],
                reverse_clip=config["REVERSE_CLIP"]
            )
        )
    elif split == "test":
        return DanceTrack(config=config, split=split, transform=transforms_for_eval())
    else:
        raise ValueError(f"Data split {split} is not supported for DanceTrack dataset.")