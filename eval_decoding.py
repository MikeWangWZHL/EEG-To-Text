import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
import re
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BertTokenizer
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive
from metrics import compute_metrics
from config import get_config


def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path='./results/temp.txt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    model.eval()  # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    # sample_count = 0

    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    with open(output_all_results_path, 'w') as f:
        # count=0
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in \
        dataloaders['test']:
            # count+=1
            # if count>5:
            #     break
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)

            if intput_noise:
                input_embeddings_batch=torch.rand_like(input_embeddings_batch)
            # target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch, skip_special_tokens = True)
            target_string = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)
            # print('target ids tensor:',target_ids_batch[0])
            # print('target ids:',target_ids_batch[0].tolist())
            # print('+' * 100)
            # print('target tokens:',target_tokens)
            # print('target string:', target_string)

            # add to list for later calculate bleu metric
            # target_tokens_list.append([target_tokens])
            target_string_list.extend(target_string)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100
            if not teacher_forcing:
                predictions = model.generate(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                                             target_ids_batch,
                                             max_length=100,
                                             num_beams=5, do_sample=False, repetition_penalty=5.0,

                                             # num_beams=5,encoder_no_repeat_ngram_size =1,
                                             # do_sample=True, top_k=15,temperature=0.5,num_return_sequences=5,
                                             # early_stopping=True

                                             )
            # predicted_string=predicted_string.squeeze()
            # print(f'predictions:{predictions}')
            # print(f'predicted_string:{predicted_string}')
            #
            # print(f'predicted_string:{predicted_string}')
            else:
                seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                                        target_ids_batch)
                logits = seq2seqLMoutput.logits  # bs*seq_len*voc_sz
                probs = logits.softmax(dim=-1)
                values, predictions = probs.topk(1)
                predictions = torch.squeeze(predictions, dim=-1)
                # print(f'predictions:{predictions} predictions shape:{predictions.shape}')
            predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True, )
            # print(f'predicted_string:{predicted_string}')

            # start = predicted_string.find("[CLS]") + len("[CLS]")
            # end = predicted_string.find("[SEP]")
            # predicted_string = predicted_string[start:end]
            # predicted_string=merge_consecutive_duplicates(predicted_string,'。')
            # predictions=tokenizer.encode(predicted_string)
            for str_id in range(len(target_string)):
                f.write(f'start################################################\n')
                f.write(f'Predicted: {predicted_string[str_id]}\n')
                f.write(f'True: {target_string[str_id]}\n')
                f.write(f'end################################################\n\n\n')
            # convert to int list
            # predictions = predictions.tolist()
            # truncated_prediction = []
            # for t in predictions:
            #     if t != tokenizer.eos_token_id:
            #         truncated_prediction.append(t)
            #     else:
            #         break
            # pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            # pred_tokens_list.append(pred_tokens)
            pred_string_list.extend(predicted_string)
            # sample_count += 1
            # print('predicted tokens:',pred_tokens)
            # print('predicted string:',predicted_string)
            # print('-' * 100)
    # print(f'pred_string_list:{pred_string_list}')
    # print(f'target_string_list:{target_string_list}')
    metrics_results=compute_metrics(pred_string_list,target_string_list)
    print(f'teacher_forcing{teacher_forcing} intput_noise{intput_noise}')
    print(metrics_results)
    print(output_all_results_path)
    print(output_all_metrics_results_path)
    with open(output_all_metrics_results_path, "w") as json_file:
        json.dump(metrics_results, json_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    home_directory = os.path.expanduser("~")
    ''' get args'''
    args = get_config('eval_decoding')

    ''' load training config'''
    training_config = json.load(open(args['config_path']))

    batch_size = 1

    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')

    dataset_setting = 'unique_sent'

    task_name = training_config['task_name']

    model_name = training_config['model_name']
    # model_name = 'BrainTranslator'
    # model_name = 'BrainTranslatorNaive'
    # teacher_forcing = True
    # {'wer': 0.7980769276618958, 'rouge1_fmeasure': 23.912235260009766, 'rouge1_precision': 24.66936492919922, 'rouge1_recall': 23.318071365356445, 'rouge2_fmeasure': 6.851282119750977, 'rouge2_precision': 6.962162017822266, 'rouge2_recall': 6.751219272613525, 'rougeL_fmeasure': 22.912235260009766, 'rougeL_precision': 23.61673355102539, 'rougeL_recall': 22.36568832397461, 'rougeLsum_fmeasure': 22.912235260009766, 'rougeLsum_precision': 23.61673355102539, 'rougeLsum_recall': 22.36568832397461, 'bleu-1': 0.23883000016212463, 'bleu-2': 0.13888777792453766, 'bleu-3': 0.0, 'bleu-4': 0.0}

    teacher_forcing = eval(args['tf'])
    intput_noise = eval(args['noise'])
    print(f'teacher_forcing{teacher_forcing} intput_noise{intput_noise}')
    output_all_results_path = (f'./results/{task_name}-{model_name}{"-teacher_forcing" if teacher_forcing else ""}{"-intput_noise" if intput_noise else ""}-all_decoding_results.txt')
    output_all_metrics_results_path = output_all_results_path.replace('txt', 'json')
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')


    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = 'datasets/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        dataset_path_task1=os.path.join(home_directory,dataset_path_task1)
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = 'datasets/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
        dataset_path_task2=os.path.join(home_directory,dataset_path_task2)
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = 'datasets/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        dataset_path_task3=os.path.join(home_directory,dataset_path_task3)
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = 'datasets/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
        dataset_path_taskNRv2=os.path.join(home_directory,dataset_path_taskNRv2)
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    print()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))

    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    if model_name == 'BrainTranslator':
        model = BrainTranslator(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    elif model_name == 'BrainTranslatorNaive':
        model = BrainTranslatorNaive(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    ''' eval '''
    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = output_all_results_path)
