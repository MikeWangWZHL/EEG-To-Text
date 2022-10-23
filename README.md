# [(AAAI 2022) Open Vocabulary EEG-To-Text Decoding and Zero-shot sentiment classification](https://arxiv.org/abs/2112.02690)
## Create Environment
run `conda env create -f environment.yml` to create the conda environment (named "EEGToText") used in our experiments.
## Download ZuCo datasets
- Download ZuCo v1.0 'Matlab files' for 'task1-SR','task2-NR','task3-TSR' from https://osf.io/q3zws/files/ under 'OSF Storage' root,  
unzip and move all `.mat` files to `/dataset/ZuCo/task1-SR/Matlab_files`,`/dataset/ZuCo/task2-NR/Matlab_files`,`/dataset/ZuCo/task3-TSR/Matlab_files` respectively.
- Download ZuCo v2.0 'Matlab files' for 'task1-NR' from https://osf.io/2urht/files/ under 'OSF Storage' root, unzip and move all `.mat` files to `/dataset/ZuCo/task2-NR-2.0/Matlab_files`.

## Preprocess datasets
run `bash ./scripts/prepare_dataset.sh` to preprocess `.mat` files and prepare sentiment labels. 

For each task, all `.mat` files will be converted into one `.pickle` file stored in `/dataset/ZuCo/<task_name>/<task_name>-dataset.pickle`. 

Sentiment dataset for ZuCo (`sentiment_labels.json`) will be stored in `/dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json`. 

Sentiment dataset for filtered Stanford Sentiment Treebank will be stored in `/dataset/stanfordsentiment/ternary_dataset.json`

## Usage Example
### Open vocabulary EEG-To-Text Decoding
To train an EEG-To-Text decoding model, run `bash ./scripts/train_decoding.sh`.

To evaluate the trained EEG-To-Text decoding model from above, run `bash ./scripts/eval_decoding.sh`.

For detailed configuration of the available arguments, please refer to function `get_config(case = 'train_decoding')` in `/config.py`

### Zero-shot sentiment classification pipeline 
We first train the decoder and the classifier individually, and then we evaluate the pipeline on ZuCo task1-SR data.

To run the whole training and evaluation process, run `bash ./scripts/train_eval_zeroshot_pipeline.sh`.

For detailed configuration of the available arguments, please refer to function `get_config(case = 'eval_sentiment')` in `/config.py`

## Citation
```
@inproceedings{wang2022open,
  title={Open vocabulary electroencephalography-to-text decoding and zero-shot sentiment classification},
  author={Wang, Zhenhailong and Ji, Heng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={5},
  pages={5350--5358},
  year={2022}
}
```
