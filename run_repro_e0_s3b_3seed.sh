#!/usr/bin/env bash

set -Eeuo pipefail

cd "$(dirname "$0")"

mkdir -p logs
mkdir -p checkpoints
mkdir -p experiments/repro_E0_S3B_3seed

MASTER_STATUS="experiments/repro_E0_S3B_3seed/run_status.tsv"

printf "method\tseed\tstage\tstatus\n" > "${MASTER_STATUS}"

echo "============================================================"
echo "E0 versus S3-B three-seed paired reproduction"
echo "Seeds: 123 456 789"
echo "Checkpoint criterion: highest validation overall IoU"
echo "GPU: CUDA_VISIBLE_DEVICES=0"
echo "============================================================"


# ------------------------------------------------------------
# Preflight checks
# ------------------------------------------------------------
for required_file in \
    train.py \
    train_s3b.py \
    test.py \
    test_s3b.py \
    lib/bounded_gated_small_refinement.py
do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: missing required file: ${required_file}"
        exit 1
    fi
done


if ! grep -q 'FIANET_SEED' train.py; then
    echo "ERROR: train.py does not support FIANET_SEED."
    exit 1
fi

if ! grep -q 'FIANET_SEED' train_s3b.py; then
    echo "ERROR: train_s3b.py does not support FIANET_SEED."
    exit 1
fi

echo "FIANET environment-seed audit: PASS"


COMMON_TRAIN_ARGS=(
    --dataset rrsisd
    --model lavt_one
    --epochs 40
    --lr 3e-5
    --wd 0.01
    --batch-size 8
    --workers 2
    --img_size 480
    --num_tmem 3
    --swin_type base
    --window12
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth
    --bert_tokenizer ./bert-base-uncased
    --ck_bert ./bert-base-uncased
    --refer_data_root /home/ubuntu/data/RRSIS-D
    --pin_mem
    --print-freq 20
)


COMMON_TEST_ARGS=(
    --dataset rrsisd
    --model lavt_one
    --split test
    --img_size 480
    --num_tmem 3
    --workers 2
    --swin_type base
    --window12
    --bert_tokenizer ./bert-base-uncased
    --ck_bert ./bert-base-uncased
    --refer_data_root /home/ubuntu/data/RRSIS-D
)


check_new_run_paths() {
    local output_dir="$1"
    local train_log="$2"
    local test_log="$3"

    if [[ -e "${output_dir}" ]]; then
        echo "ERROR: output directory already exists:"
        echo "  ${output_dir}"
        echo "Rename or remove it before launching this fresh run."
        exit 1
    fi

    if [[ -e "${train_log}" ]]; then
        echo "ERROR: training log already exists:"
        echo "  ${train_log}"
        echo "Rename or remove it before launching this fresh run."
        exit 1
    fi

    if [[ -e "${test_log}" ]]; then
        echo "ERROR: test log already exists:"
        echo "  ${test_log}"
        echo "Rename or remove it before launching this fresh run."
        exit 1
    fi
}


run_e0() {
    local seed="$1"

    local model_id="FIANet_E0_amp_bs8_full40_seed${seed}"
    local output_dir="checkpoints/${model_id}"
    local train_log="logs/train_${model_id}.log"
    local test_log="logs/test_${model_id}.log"
    local checkpoint="${output_dir}/model_best_${model_id}.pth"

    check_new_run_paths \
        "${output_dir}" \
        "${train_log}" \
        "${test_log}"

    echo
    echo "============================================================"
    echo "START E0 seed=${seed}"
    echo "Training log: ${train_log}"
    echo "Output directory: ${output_dir}"
    echo "============================================================"

    printf "E0\t%s\ttrain\trunning\n" "${seed}" >> "${MASTER_STATUS}"

    env \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONHASHSEED="${seed}" \
        FIANET_SEED="${seed}" \
        python -u train.py \
        "${COMMON_TRAIN_ARGS[@]}" \
        --model_id "${model_id}" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${train_log}"

    if [[ ! -f "${checkpoint}" ]]; then
        echo "ERROR: E0 checkpoint not found:"
        echo "  ${checkpoint}"
        printf "E0\t%s\ttrain\tcheckpoint_missing\n" "${seed}" >> "${MASTER_STATUS}"
        exit 1
    fi

    printf "E0\t%s\ttrain\tcompleted\n" "${seed}" >> "${MASTER_STATUS}"

    echo
    echo "============================================================"
    echo "TEST E0 seed=${seed}"
    echo "Checkpoint: ${checkpoint}"
    echo "Test log: ${test_log}"
    echo "============================================================"

    printf "E0\t%s\ttest\trunning\n" "${seed}" >> "${MASTER_STATUS}"

    env \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONHASHSEED="${seed}" \
        FIANET_SEED="${seed}" \
        python -u test.py \
        "${COMMON_TEST_ARGS[@]}" \
        --resume "${checkpoint}" \
        2>&1 | tee "${test_log}"

    printf "E0\t%s\ttest\tcompleted\n" "${seed}" >> "${MASTER_STATUS}"

    echo "FINISHED E0 seed=${seed}"
}


run_s3b() {
    local seed="$1"

    local model_id="FIANet_S3B_bounded_gated_refine_amp_bs8_full40_seed${seed}"
    local output_dir="checkpoints/${model_id}"
    local train_log="logs/train_${model_id}.log"
    local test_log="logs/test_${model_id}.log"
    local checkpoint="${output_dir}/model_best_${model_id}.pth"

    check_new_run_paths \
        "${output_dir}" \
        "${train_log}" \
        "${test_log}"

    echo
    echo "============================================================"
    echo "START S3-B seed=${seed}"
    echo "Training log: ${train_log}"
    echo "Output directory: ${output_dir}"
    echo "============================================================"

    printf "S3B\t%s\ttrain\trunning\n" "${seed}" >> "${MASTER_STATUS}"

    env \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONHASHSEED="${seed}" \
        FIANET_SEED="${seed}" \
        python -u train_s3b.py \
        "${COMMON_TRAIN_ARGS[@]}" \
        --model_id "${model_id}" \
        --output-dir "${output_dir}" \
        --use-small-refine \
        --small-project-channels 16 \
        --small-refine-channels 32 \
        2>&1 | tee "${train_log}"

    if [[ ! -f "${checkpoint}" ]]; then
        echo "ERROR: S3-B checkpoint not found:"
        echo "  ${checkpoint}"
        printf "S3B\t%s\ttrain\tcheckpoint_missing\n" "${seed}" >> "${MASTER_STATUS}"
        exit 1
    fi

    printf "S3B\t%s\ttrain\tcompleted\n" "${seed}" >> "${MASTER_STATUS}"

    echo
    echo "============================================================"
    echo "TEST S3-B seed=${seed}"
    echo "Checkpoint: ${checkpoint}"
    echo "Test log: ${test_log}"
    echo "============================================================"

    printf "S3B\t%s\ttest\trunning\n" "${seed}" >> "${MASTER_STATUS}"

    env \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONHASHSEED="${seed}" \
        FIANET_SEED="${seed}" \
        python -u test_s3b.py \
        "${COMMON_TEST_ARGS[@]}" \
        --resume "${checkpoint}" \
        --use-small-refine \
        --small-project-channels 16 \
        --small-refine-channels 32 \
        2>&1 | tee "${test_log}"

    printf "S3B\t%s\ttest\tcompleted\n" "${seed}" >> "${MASTER_STATUS}"

    echo "FINISHED S3-B seed=${seed}"
}


# Paired execution keeps the comparison organized by seed.
for seed in 123 456 789
do
    run_e0 "${seed}"
    run_s3b "${seed}"
done


echo
echo "============================================================"
echo "ALL E0 AND S3-B THREE-SEED RUNS COMPLETED"
echo "Status file: ${MASTER_STATUS}"
echo "============================================================"
