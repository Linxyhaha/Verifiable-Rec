#!/bin/bash
# Single-GPU driver for VRec, full 3-stage pipeline:
#   pretrain_LLM -> verifier_pretrain (MoV, LLM frozen) -> finetune -> eval -> calc.
# The repo's scripts/run.sh is an 8-GPU template and is NOT runnable verbatim
# (passes --beta/--gamma which finetune/train.py does not accept -> here we use
# --v_weight/--mono_weight). This driver runs all stages on 1 GPU.
#
# Usage:
#   BASE_QWEN=/path/to/Qwen2.5-1.5B SAMPLE=-1 bash scripts/run_single.sh MicroLens
#   SMOKE=1 bash scripts/run_single.sh MicroLens          # tiny end-to-end check
set -e

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

category=${1:-MicroLens}
BASE_QWEN=${BASE_QWEN:-/data/repos/fbsource/Qwen2.5-1.5B}

N_THOUGHT=${N_THOUGHT:-1}
N_CLUSTER=${N_CLUSTER:-25}
N_VERIFIER=${N_VERIFIER:-2}
BATCH=${BATCH:-128}

# SMOKE mode: tiny sample + 1 epoch to validate the full pipeline quickly.
if [ "${SMOKE:-0}" = "1" ]; then
  SAMPLE=${SAMPLE:-64}; PRE_EPOCHS=${PRE_EPOCHS:-1}; VPRE_EPOCHS=${VPRE_EPOCHS:-1}; FT_EPOCHS=${FT_EPOCHS:-1}
else
  SAMPLE=${SAMPLE:--1}; PRE_EPOCHS=${PRE_EPOCHS:-3}; VPRE_EPOCHS=${VPRE_EPOCHS:-3}; FT_EPOCHS=${FT_EPOCHS:-3}
fi
PRETRAIN_MBS=${PRETRAIN_MBS:-16}
FT_MBS=${FT_MBS:-4}
EVAL_BS=${EVAL_BS:-8}

# Resolve to the code/ directory (parent of scripts/).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TORCHRUN=.venv/bin/torchrun

DATA=../data/${category}
train_file=${DATA}/train/${category}.csv
eval_file=${DATA}/valid/${category}.csv
test_file=${DATA}/${TDIR:-test}/${category}.csv
info_file=${DATA}/info/${category}.txt
asin2catid_file=${DATA}/info/${category}_asin2cat_qwen_cluster_${N_CLUSTER}.npy
asin2clusterid_file=${DATA}/info/${category}_asin2cat_cf_cluster_${N_CLUSTER}.npy

OUT=./output_dir/${category}
PRE=${OUT}/pretrain_${N_THOUGHT}T
VPRE=${OUT}/verifier_pretrain_${N_VERIFIER}v_${N_THOUGHT}T
FT=${OUT}/vrec_${N_VERIFIER}v_${N_THOUGHT}T
mkdir -p "$PRE" "$VPRE" "$FT"
VPRE_LR=${VPRE_LR:-1e-4}

MODEL_CLASS=LatentModelwithVerifier_RATT_MS_MoV

echo "=============================================="
echo " category=$category  SAMPLE=$SAMPLE  n_thought=$N_THOUGHT"
echo " pre_epochs=$PRE_EPOCHS  ft_epochs=$FT_EPOCHS  base=$BASE_QWEN"
echo "=============================================="

if [ "${SKIP_PRETRAIN:-0}" != "1" ]; then
echo ">>> [1/5] pretrain_LLM"
$TORCHRUN --standalone --nproc_per_node=1 src/pretrain_LLM/train.py \
    --base_model "$BASE_QWEN" \
    --train_file "$train_file" --eval_file "$eval_file" \
    --output_dir "$PRE" --category "$category" --sample "$SAMPLE" \
    --n_thought "$N_THOUGHT" \
    --num_epochs "$PRE_EPOCHS" \
    --micro_batch_size "$PRETRAIN_MBS" --batch_size "$BATCH"
else echo ">>> [1/5] pretrain_LLM  (SKIPPED, reuse $PRE)"; fi

if [ "${SKIP_VERIFIER:-0}" != "1" ]; then
echo ">>> [2/5] verifier pretraining (MoV, LLM frozen)"
$TORCHRUN --standalone --nproc_per_node=1 src/pretrain_verifier/train.py \
    --base_model "$PRE" --model_class "$MODEL_CLASS" \
    --train_file "$train_file" --eval_file "$eval_file" \
    --asin2catid_file "$asin2catid_file" --asin2clusterid_file "$asin2clusterid_file" \
    --output_dir "$VPRE" --category "$category" --sample "$SAMPLE" \
    --n_thought "$N_THOUGHT" --num_verifier "$N_VERIFIER" \
    --v_weight 1 --mono_weight 0 \
    --learning_rate "$VPRE_LR" \
    --micro_batch_size "$FT_MBS" --batch_size "$BATCH" --num_epochs "$VPRE_EPOCHS"
else echo ">>> [2/5] verifier pretraining  (SKIPPED, reuse $VPRE)"; fi

echo ">>> [3/5] finetune (VRec MoV)"
$TORCHRUN --standalone --nproc_per_node=1 src/finetune/train.py \
    --base_model "$VPRE" --model_class "$MODEL_CLASS" \
    --train_file "$train_file" --eval_file "$eval_file" \
    --asin2catid_file "$asin2catid_file" --asin2clusterid_file "$asin2clusterid_file" \
    --output_dir "$FT" --category "$category" --sample "$SAMPLE" \
    --n_thought "$N_THOUGHT" --num_verifier "$N_VERIFIER" \
    --v_weight 1 --mono_weight 1 \
    --learning_rate 5e-5 \
    --micro_batch_size "$FT_MBS" --batch_size "$BATCH" --num_epochs "$FT_EPOCHS"

echo ">>> [4/5] eval"
$PY src/finetune/eval.py --batch_size "$EVAL_BS" \
    --base_model "$FT" --sample "$SAMPLE" --model_class "$MODEL_CLASS" \
    --info_file "$info_file" --category "$category" \
    --test_data_path "$test_file" \
    --asin2catid_file "$asin2catid_file" --asin2clusterid_file "$asin2clusterid_file" \
    --n_thought "$N_THOUGHT" \
    --result_json_data "${FT}/result.json"

echo ">>> [5/5] calc (NDCG@[1,3,5,10,20], HR@[1,3,5,10,20], CC)"
$PY src/utils/calc.py --path "${FT}/result.json" --item_path "$info_file"
