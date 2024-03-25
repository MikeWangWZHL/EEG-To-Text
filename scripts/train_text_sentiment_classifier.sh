python3 train_sentiment_textbased.py \
    --dataset_name SST \
    --model_name pretrain_Bart \
    --num_epoch 20 \
    -lr 0.0001 \
    -b 32 \
    -s ./checkpoints/text_sentiment_classifier \
    -cuda cuda:0