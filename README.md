Wave-MambaTrack
train：
nohup python -m torch.distributed.run --nproc_per_node=2 main.py     --use-distributed     --config-path ./configs/train_dancetrack.yaml     --outputs-dir Outputs-root     --batch-size 1     --data-root Datasets-root     --use-checkpoint     > train_log.txt 2>&1 &
eval:
python main.py    --config-path ./configs/train_dancetrack.yaml    --mode submit    --submit-dir ./outputs/memotr_dancetrack/    --submit-model checkpoint-root    --data-root Datasets-root 
