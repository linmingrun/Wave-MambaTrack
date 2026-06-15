import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def draw_tracks_from_mot_txt(
        txt_path,
        save_path="tracks_gt_style.png",
        img_w=1920,
        img_h=1080,
        min_track_len=5,
        show_tracks=38,
        random_seed=42
):
    np.random.seed(random_seed)

    tracks = defaultdict(list)

    # =====================
    # 读取 txt
    # =====================
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue

            frame = int(float(p[0]))
            tid = int(float(p[1]))

            x = float(p[2])
            y = float(p[3])
            w = float(p[4])
            h = float(p[5])

            # 使用 bbox 中心点
            cx = x + w / 2.0
            cy = y + h / 2.0

            tracks[tid].append((frame, cx, cy))

    # =====================
    # 过滤短轨迹
    # =====================
    valid = []
    for tid, pts in tracks.items():
        if len(pts) >= min_track_len:
            pts = sorted(pts, key=lambda x: x[0])  # 按 frame 排序
            valid.append((tid, pts, len(pts)))

    if len(valid) == 0:
        print("没有满足 min_track_len 的轨迹。")
        return

    # 按轨迹长度从大到小排序
    valid.sort(key=lambda x: x[2], reverse=True)

    lengths = np.array([x[2] for x in valid])

    # =====================
    # 分层采样：长 / 中 / 短
    # =====================
    q1 = np.percentile(lengths, 30)
    q2 = np.percentile(lengths, 70)

    long_tracks = []
    mid_tracks = []
    short_tracks = []

    for item in valid:
        l = item[2]
        if l >= q2:
            long_tracks.append(item)
        elif l >= q1:
            mid_tracks.append(item)
        else:
            short_tracks.append(item)

    n_long = int(show_tracks * 0.3)
    n_mid = int(show_tracks * 0.4)
    n_short = show_tracks - n_long - n_mid

    selected = []

    # 长轨迹：优先取前 n_long 条
    selected += long_tracks[:min(n_long, len(long_tracks))]

    # 中轨迹：随机抽样（修复点：不能直接对 list of tuples 用 np.random.choice）
    if len(mid_tracks) > 0:
        idx = np.random.choice(
            len(mid_tracks),
            size=min(n_mid, len(mid_tracks)),
            replace=False
        )
        selected += [mid_tracks[i] for i in idx]

    # 短轨迹：随机抽样
    if len(short_tracks) > 0:
        idx = np.random.choice(
            len(short_tracks),
            size=min(n_short, len(short_tracks)),
            replace=False
        )
        selected += [short_tracks[i] for i in idx]

    # =====================
    # 绘图
    # =====================
    plt.figure(figsize=(13, 9))
    ax = plt.gca()

    colors = [
        "black",
        "#ff7f0e",
        "#2ca02c",
        "#1f77b4",
        "#d62728",
        "#9467bd"
    ]

    for i, item in enumerate(selected):
        tid = item[0]
        pts = item[1]

        x = [p[1] for p in pts]
        y = [p[2] for p in pts]

        # 截断超长轨迹，让视觉效果更接近参考图
        if len(x) > 180:
            start = np.random.randint(0, len(x) - 180)
            x = x[start:start + 180]
            y = y[start:start + 180]

        # 前几个最长轨迹用黑色
        if i < 5:
            color = "black"
        else:
            color = colors[i % len(colors)]

        lw = np.clip(len(x) / 70.0, 1.8, 3.0)

        ax.plot(
            x, y,
            color=color,
            linewidth=lw,
            alpha=0.95,
            solid_capstyle="round"
        )

    # =====================
    # 坐标轴设置
    # =====================
    ax.set_xlim(0, img_w)
    ax.set_ylim(0, img_h)

    ax.set_xticks([0, 480, 960, 1440, 1920])
    ax.set_yticks([0, 270, 540, 810, 1080])

    ax.set_xlabel("Image length/pixel", fontsize=18)
    ax.set_ylabel("Image width/pixel", fontsize=18)

    ax.tick_params(labelsize=14, width=2, length=8)

    for s in ax.spines.values():
        s.set_linewidth(2)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("总轨迹:", len(valid))
    print("显示轨迹:", len(selected))
    print("保存到:", save_path)


# =====================
# 直接运行
# =====================
txt_path = "/nfs_B/LiaoYangHao/MeMOTR/outputs/GrassTrack/TransTrack/seq9.txt"

draw_tracks_from_mot_txt(
    txt_path=txt_path,
    save_path="track_plot.png",
    show_tracks=38,
    min_track_len=5
)