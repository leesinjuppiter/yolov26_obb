#!/usr/bin/env bash
# 依次验证 OBB 推理、训练，并用相同的通用预训练权重训练 W0。
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  printf '用法: bash %s 工作区 运行标识\n' "$0" >&2
  exit 2
fi
workspace=$(realpath "$1")
run_id=$2
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]
data="$workspace/dataset"
runs="$workspace/runs/yolo26_obb_al/$run_id"
weight="$workspace/weights/yolo26n-obb.pt"
py="$HOME/.venvs/yolo26-autodl/bin/python"
yolo="$HOME/.venvs/yolo26-autodl/bin/yolo"
[[ -x "$py" && -x "$yolo" ]]
mkdir -p "$runs" "$workspace/weights" "$workspace/yolo26-config"
export YOLO_CONFIG_DIR="$workspace/yolo26-config"
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=2
cd "$workspace"

stage=preflight
status() { printf '%s %s\n' "$(date -Is)" "$1" > "$runs/status.txt"; }
trap 'rc=$?; status "失败 阶段=$stage 退出码=$rc"; exit "$rc"' ERR
status "$stage"
for role in L0_initial_train V_validation T_independent_test; do
  [[ -d "$data/$role/images" && -d "$data/$role/labels" ]]
done
sample="$data/L0_initial_train/images/d4911_l0_000001.jpg"
src="$data/dataset4911_yolo26obb_al.yaml"
cfg="$runs/dataset_linux.yaml"
test -s "$sample" && test -s "$src"
"$py" -B - "$src" "$data" "$cfg" <<'PY'
import sys
from pathlib import Path
import torch
import ultralytics
import yaml

src, data, cfg = map(Path, sys.argv[1:])
assert ultralytics.__version__ == '8.4.115'
assert torch.cuda.is_available()
x = torch.ones(1, device='cuda:0')
assert (x + 1).item() == 2
print('Environment:', ultralytics.__version__, torch.__version__, torch.cuda.get_device_name(0))
spec = yaml.safe_load(src.read_text(encoding='utf-8'))
roles = {'train': 'L0_initial_train/images', 'val': 'V_validation/images',
         'test': 'T_independent_test/images'}
assert all(spec.get(k) == v for k, v in roles.items())
assert list(spec['names'].values()) == ['coal', 'rockbolt', 'mesh', 'stick', 'board', 'cap', 'bottle']
spec['path'] = str(data.resolve())
cfg.write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
PY

stage=pretrained_weight
status "$stage"
if [[ ! -s "$weight" ]]; then
  part=$(mktemp "$workspace/weights/.yolo26n-obb.XXXXXXXX")
  trap 'rm -f -- "$part"' EXIT
  curl --fail --location --retry 2 --connect-timeout 15 --max-time 180 \
    'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.pt' -o "$part"
  [[ "$(stat -c %s "$part")" -gt 100000 ]]
  mv -- "$part" "$weight"
  trap - EXIT
fi
sha256sum -- "$weight" > "$runs/pretrained.sha256"

stage=smoke_predict
status "$stage"
"$yolo" obb predict model="$weight" source="$sample" device=0 imgsz=640 \
  project="$runs" name=smoke_predict save=True
test -s "$runs/smoke_predict/d4911_l0_000001.jpg"

stage=smoke_train
status "$stage"
"$yolo" obb train model="$weight" data="$cfg" device=0 imgsz=640 batch=4 workers=2 \
  epochs=1 fraction=0.05 seed=26 project="$runs" name=smoke_train plots=False
test -s "$runs/smoke_train/weights/best.pt"
"$py" -B - "$runs/smoke_train/results.csv" <<'PY'
import csv
import math
import sys

with open(sys.argv[1], newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f, skipinitialspace=True))
assert len(rows) == 1, 'Expected exactly one smoke-training epoch'
values = {k.strip(): float(v) for k, v in rows[0].items() if v.strip()}
assert values and all(math.isfinite(v) for v in values.values())
assert any(k.startswith('metrics/mAP') for k in values), 'OBB validation metrics missing'
print('Smoke training and validation completed with finite metrics.')
PY
touch "$runs/smoke_passed"

stage=w0_training
status "$stage"
"$yolo" obb train model="$weight" data="$cfg" device=0 imgsz=640 batch=4 workers=2 \
  epochs=100 patience=20 seed=26 project="$runs" name=w0_initial
test -s "$runs/w0_initial/weights/best.pt"
sha256sum -- "$runs/w0_initial/weights/best.pt" > "$runs/w0_best.sha256"
status w0_complete
