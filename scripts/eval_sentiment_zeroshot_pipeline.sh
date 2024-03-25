python3 eval_sentiment.py --model_name ZeroShotSentimentDiscovery \
    --decoder_checkpoint_path ./checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --classifier_checkpoint_path ./checkpoints/text_sentiment_classifier/best/Textbased_StanfordSentitmentTreeband_pretrain_Bart_b32_20_0.0001.pt \
    --decoder_config_path ./config/decoding/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    --classifier_config_path ./config/text_sentiment_classifier/Textbased_StanfordSentitmentTreeband_pretrain_Bart_b32_20_0.0001.json \
    --cuda cuda:0