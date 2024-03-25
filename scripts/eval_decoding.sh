python3 eval_decoding.py \
    --checkpoint_path ./checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --config_path ./config/decoding/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    -cuda cuda:0 -tf True -n True

python3 eval_decoding.py \
    --checkpoint_path ./checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --config_path ./config/decoding/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    -cuda cuda:0 -tf True -n False

python3 eval_decoding.py \
    --checkpoint_path ./checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --config_path ./config/decoding/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    -cuda cuda:0 -tf False -n True

python3 eval_decoding.py \
    --checkpoint_path ./checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --config_path ./config/decoding/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    -cuda cuda:0 -tf False -n False


