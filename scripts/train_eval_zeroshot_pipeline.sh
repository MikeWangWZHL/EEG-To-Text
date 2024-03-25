
echo "###################################"
echo "Training decoder: BART, task1-SR..."
echo "###################################"
echo ""
python3 train_decoding.py --model_name BrainTranslator \
    --task_name task1 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 20 \
    --num_epoch_step2 30 \
    -lr1 0.00005 \
    -lr2 0.0000005 \
    -b 32 \
    -s ./checkpoints/decoding \
    -cuda cuda:0

echo "###################################"
echo "Training classifier: BART, filtered Stanford Sentiment Treebank..."
echo "###################################"
echo ""
python3 train_sentiment_textbased.py \
    --dataset_name SST \
    --model_name pretrain_Bart \
    --num_epoch 20 \
    -lr 0.0001 \
    -b 32 \
    -s ./checkpoints/text_sentiment_classifier \
    -cuda cuda:0

echo "###################################"
echo "Evaluating Zero-shot pipeline: DEC(BART) + CLS(BART)"
echo "###################################"
echo ""
python3 eval_sentiment.py --model_name ZeroShotSentimentDiscovery \
    --decoder_checkpoint_path ./checkpoints/decoding/best/task1_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --classifier_checkpoint_path ./checkpoints/text_sentiment_classifier/best/Textbased_StanfordSentitmentTreeband_pretrain_Bart_b32_20_0.0001.pt \
    --decoder_config_path ./config/decoding/task1_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    --classifier_config_path ./config/text_sentiment_classifier/Textbased_StanfordSentitmentTreeband_pretrain_Bart_b32_20_0.0001.json \
    --cuda cuda:0