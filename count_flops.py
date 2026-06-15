import torch
from thop import profile
import yaml

from models.memotr import build
from structures.track_instances import TrackInstances
from utils.nested_tensor import NestedTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("/nfs_B/LiaoYangHao/MOTIP/configs/r50_deformable_detr_motip_dancetrack.yaml", "r") as f:
    config = yaml.safe_load(f)

config["DATASET"] = config["DATASETS"][0]                 # DanceTrack
config["NUM_DET_QUERIES"] = config["DETR_NUM_QUERIES"]    # 300
config["NUM_FEATURE_LEVELS"] = config["DETR_NUM_FEATURE_LEVELS"]  # 4
config["HIDDEN_DIM"] = config["DETR_HIDDEN_DIM"]          # 256
config["FFN_DIM"] = config["DETR_DIM_FEEDFORWARD"]        # 1024
config["DROPOUT"] = config["DETR_DROPOUT"]                # 0.0
config["USE_CHECKPOINT"] = False                          # 统计 FLOPs 时建议关掉
config["CHECKPOINT_LEVEL"] = 0
config["USE_DAB"] = False                                 # 先按你的当前模型设定
config["VISUALIZE"] = False
model = build(config).to(device)
model.eval()

B = 1
img = torch.randn(B, 3, 800, 1333, device=device)

dummy = TrackInstances()
dummy.ref_pts = torch.zeros((1, 4), device=device)

query_dim = model.hidden_dim if model.use_dab else model.hidden_dim * 2
dummy.query_embed = torch.zeros((1, query_dim), device=device)

tracks = [dummy]

class Wrapper(torch.nn.Module):
    def __init__(self, model, tracks):
        super().__init__()
        self.model = model
        self.tracks = tracks

    def forward(self, x):
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)
        nested = NestedTensor(x, mask)
        return self.model(nested, self.tracks)["pred_logits"]

wrapped = Wrapper(model, tracks).to(device)

flops, params = profile(
    wrapped,
    inputs=(img,),
    verbose=False
)

print(f"Params: {params / 1e6:.2f} M")
print(f"FLOPs : {flops / 1e9:.2f} G")