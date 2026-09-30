#!/usr/bin/env bash
#SBATCH --job-name=btpn_vmf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=40:00:00
#SBATCH --output=%x_%j.out
#
# Retrain BTPN with the vMF rotation likelihood.
#
#   STAGE=foundation   train the kinematic foundation model (frozen prior)
#   STAGE=supervised   train the visual BTPN on top of it (reuses the released SSL encoder),
#                      then evaluate on Dataset A
#   SMOKE=1            1 epoch per stage, 1 MC sample: checks paths/configs end to end
#
#   sbatch --export=ALL,STAGE=foundation scripts/aire/retrain_vmf_fix.slurm
set -euo pipefail

R="${BTPN_ROOT:-$SCRATCH/btpn-vmf-fix/BTPN}"
PY="${BTPN_PY:-/users/sc20osc/.conda/envs/endovis-keypoints/bin/python}"
STAGE="${STAGE:?set STAGE=foundation|supervised}"
SMOKE="${SMOKE:-0}"
RUN="${RUN:-vmf_fix}"
[ "$SMOKE" = "1" ] && RUN="${RUN}_smoke"
OUT="$R/outputs/$RUN"
cd "$R"
# The package is not pip-installed on AIRE; scripts/*.py import `btpn` from the repo root.
export PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0

echo "[job] id=${SLURM_JOB_ID:-NA} node=$(hostname) stage=$STAGE smoke=$SMOKE run=$RUN $(date)"
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('[job] torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0))"

EPOCHS=()
MC=20
if [ "$SMOKE" = "1" ]; then EPOCHS=(--epochs 1); MC=1; fi

export PYTHONUNBUFFERED=1

if [ "$STAGE" = "foundation" ]; then
  "$PY" -u scripts/train.py --stage foundation --config configs/kinematic_foundation.yaml \
    --paths configs/paths.yaml --output-dir "$OUT/foundation" "${EPOCHS[@]}"
  test -f "$OUT/foundation/checkpoints/best_model.pt"
  echo "[job] foundation done $(date)"
fi

if [ "$STAGE" = "supervised" ]; then
  KFM="$OUT/foundation/checkpoints/best_model.pt"
  test -f "$KFM" || { echo "[job] ERROR missing $KFM"; exit 1; }
  CFG="$OUT/btpn_vmf_fix.yaml"
  cp configs/btpn.yaml "$CFG"
  printf '\nkinematic_checkpoint: "%s"\n' "$KFM" >> "$CFG"

  "$PY" -u scripts/train.py --stage supervised --config "$CFG" \
    --paths configs/paths.yaml --output-dir "$OUT/btpn" \
    --stage1-checkpoint checkpoints/btpn_ssl.pt "${EPOCHS[@]}"
  CKPT="$OUT/btpn/stage2/checkpoints/best_model.pt"
  test -f "$CKPT"
  echo "[job] supervised done $(date)"

  "$PY" -u scripts/evaluate.py --checkpoint "$CKPT" --config "$CFG" --dataset A \
    --stage btpn --mc-samples "$MC" --output-dir "$OUT/eval"
  "$PY" - "$OUT/eval/evaluation_data_A.npz" <<'EOF'
import sys, numpy as np
d = np.load(sys.argv[1])
k = d["kappa_quaternion"].reshape(-1)
print(f"[job] kappa percentiles 1/25/50/75/99: {np.percentile(k, [1, 25, 50, 75, 99]).round(2)}")
EOF
  echo "[job] DONE $(date)"
fi
