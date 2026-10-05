#!/bin/bash
#SBATCH --job-name=llmpt_pipeline
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=00:58:00
#
# Whole pipeline on one node with two GPUs, as two lanes that run side by side:
#   GPU 0: SFT -> eval(SFT) -> DPO pairs + DPO -> eval(DPO)
#   GPU 1: eval(base) -> [wait for SFT] -> GRPO -> eval(GRPO)
# Every stage is skipped if its output already exists, so a resubmission resumes.
#
# Usage, from the repository root:
#   WORK=/path/to/work/dir sbatch slurm/pipeline.sbatch
# WORK holds venv/, hf/ (Hugging Face cache) and receives ckpt/ and logs/.
# Optional: RES (results dir), CK (checkpoint dir), *_SET (config overrides for quick runs).
set -uo pipefail
: "${WORK:?set WORK to the working directory}"
source "$WORK/venv/bin/activate"
export HF_HOME="$WORK/hf" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR="$WORK/triton" XDG_CACHE_HOME="$WORK/xdg/cache"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
BASE=${BASE:-Qwen/Qwen2.5-0.5B}
RES=${RES:-results}
CK=${CK:-$WORK/ckpt}
LOGS="$WORK/logs"
mkdir -p "$RES" "$CK" "$LOGS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -m pytest -q tests || exit 1

# run <marker> <command...>: skip if <marker> exists, record a FAILED marker on error
run() {
  local marker=$1; shift
  if [ -e "$marker" ]; then echo "skip ($marker exists)"; return 0; fi
  echo "[$(date +%T)] start: $*"
  local t0=$SECONDS
  "$@" || { mkdir -p "$(dirname "$marker")"; touch "$marker.FAILED"; echo "[$(date +%T)] FAILED: $*"; return 1; }
  echo "[$(date +%T)] done in $((SECONDS - t0)) s: $*"
}

lane_a() {
  export CUDA_VISIBLE_DEVICES=0
  run "$CK/sft/DONE" python -m src.sft --config configs/sft.yaml --model "$BASE" --out "$CK/sft" \
      --results "$RES" --set ${SFT_SET:-} &&
  run "$RES/eval_sft.json" python -m src.evaluate --config configs/eval.yaml --model "$CK/sft" --name sft \
      --results "$RES" --set ${EVAL_SET:-} &&
  run "$CK/dpo/DONE" python -m src.dpo --config configs/dpo.yaml --model "$CK/sft" --out "$CK/dpo" \
      --results "$RES" --set ${DPO_SET:-} &&
  run "$RES/eval_dpo.json" python -m src.evaluate --config configs/eval.yaml --model "$CK/dpo" --name dpo \
      --results "$RES" --set ${EVAL_SET:-}
}

lane_b() {
  export CUDA_VISIBLE_DEVICES=1
  run "$RES/eval_base.json" python -m src.evaluate --config configs/eval.yaml --model "$BASE" --name base \
      --results "$RES" --set ${EVAL_SET:-}
  while [ ! -e "$CK/sft/DONE" ]; do
    [ -e "$CK/sft/DONE.FAILED" ] && { echo "SFT failed, skipping GRPO"; return 1; }
    sleep 15
  done
  run "$CK/grpo/DONE" python -m src.grpo --config configs/grpo.yaml --model "$CK/sft" --out "$CK/grpo" \
      --results "$RES" --set ${GRPO_SET:-} &&
  run "$RES/eval_grpo.json" python -m src.evaluate --config configs/eval.yaml --model "$CK/grpo" --name grpo \
      --results "$RES" --set ${EVAL_SET:-}
}

JOB=${SLURM_JOB_ID:-local}
lane_a > "$LOGS/lane_a_$JOB.log" 2>&1 &
lane_b > "$LOGS/lane_b_$JOB.log" 2>&1 &
wait
python -m src.report "$RES"
echo "pipeline finished"
