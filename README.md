# Wave-MambaTrack

Official implementation of **Wave-MambaTrack** for Multi-Object Tracking.

---

## Environment Setup

Create the environment and install dependencies:

```bash
conda create -n wavemambatrack python=3.10
conda activate wavemambatrack

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib pyyaml scipy tqdm tensorboard

pip install opencv-python
cd ./models/ops/
sh make.sh

Install Mamba State Space Model (mamba-ssm):
pip install causal-conv1d
pip install mamba-ssm
```

---

## Training

Launch distributed training:

```bash
nohup python -m torch.distributed.run \
    --nproc_per_node=2 \
    main.py \
    --use-distributed \
    --config-path ./configs/train_dancetrack.yaml \
    --outputs-dir Outputs-root \
    --batch-size 1 \
    --data-root Datasets-root \
    --use-checkpoint \
> train_log.txt 2>&1 &
```

### Arguments

| Argument           | Description                |
| ------------------ | -------------------------- |
| `--nproc_per_node` | Number of GPUs             |
| `--outputs-dir`    | Training output directory  |
| `--batch-size`     | Batch size per GPU         |
| `--data-root`      | Dataset root path          |
| `--use-checkpoint` | Enable gradient checkpoint |

Training logs:

```bash
tail -f train_log.txt
```

---

## Evaluation / Submission

Generate tracking results using a trained checkpoint:

```bash
python main.py \
    --config-path ./configs/train_dancetrack.yaml \
    --mode submit \
    --submit-dir ./outputs/wavemambatrack/ \
    --submit-model checkpoint-root \
    --data-root Datasets-root
```

### Arguments

| Argument         | Description        |
| ---------------- | ------------------ |
| `--submit-dir`   | Output directory   |
| `--submit-model` | Trained checkpoint |
| `--data-root`    | Dataset path       |

---

