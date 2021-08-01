python3 eval_decoding.py \
    --checkpoint_path ./checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --config_path /shared/nas/data/m1/wangz3/SAO_project/AAAI_submission_code/config/decoding/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    -cuda cuda:0