import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer
from tqdm import tqdm
from fuzzy_match import match
from fuzzy_match import algorithims


def get_SST_dataset(SST_dir_path, ZuCo_used_sentences, ZUCO_SENTIMENT_LABELS):
    
    def get_sentiment_label_dict(SST_dictionary_file_path):
        '''
            return {phrase_id:sentiment_score(0-1)}
        '''
        ret_dict = {}
        with open(SST_dictionary_file_path) as f:
            for line in f:
                if line.startswith('phrase'):
                    continue
                else:
                    phrase_id = int(line.split('|')[0])
                    label = float(line.split('|')[1].strip())
                    assert phrase_id not in ret_dict
                    ret_dict[phrase_id] = label
        return ret_dict

    def get_phrasestr_phrase_dict(SST_dictionary_file_path):
        '''
            return {phrase_str: phrase_id}
        '''
        ret_dict = {}
        with open(SST_dictionary_file_path) as f:
            for line in f:
                phrase_str = line.split('|')[0]
                phrase_id = int(line.split('|')[1].strip())
                assert phrase_str not in ret_dict
                ret_dict[phrase_str] = phrase_id
        return ret_dict

    def get_sentence_label_dict(SST_sentences_file_path, SST_labels_file_path, SST_dictionary_file_path):
        '''
            return {sentence_str:label(0-1)}
        '''
        phraseID_2_label = get_sentiment_label_dict(SST_labels_file_path)
        phraseStr_2_phraseID = get_phrasestr_phrase_dict(SST_dictionary_file_path)

        sentence_2_label_all = {}
        sentence_2_label_ternary = {}
        with open(SST_sentences_file_path) as f:
            for line in f:
                if line.startswith('sentence_index'):
                    continue
                else:
                    parsed_line = line.split('\t')
                    assert len(parsed_line) == 2
                    sentence = parsed_line[1].strip()
                    # convert -LRB- to (, -RRB- to ):
                    sentence = sentence.replace('-LRB-','(').replace('-RRB-',')').replace('Ã©','é')
                    if sentence not in phraseStr_2_phraseID:
                        # print(f'[ERROR]sentence-phrase match not found in dictionary, skipped: {sentence}')
                        # print()
                        continue
                    sent_phrase_id = phraseStr_2_phraseID[sentence]
                    label = phraseID_2_label[sent_phrase_id]
                    
                    # add to all dict
                    if sentence not in sentence_2_label_all:
                        sentence_2_label_all[sentence] = label

                    # add to ternary dict
                    if sentence not in sentence_2_label_ternary:
                        if label<=0.2:
                            label = 0
                            sentence_2_label_ternary[sentence] = label
                        elif (label > 0.4) and (label<=0.6): 
                            label = 1
                            sentence_2_label_ternary[sentence] = label
                        elif label>0.8:
                            label = 2
                            sentence_2_label_ternary[sentence] = label

        return sentence_2_label_all, sentence_2_label_ternary


    SST_sentences_file_path = os.path.join(SST_dir_path,'datasetSentences.txt')
    if not os.path.isfile(SST_sentences_file_path):
        print(f'NOT FOUND file: {SST_sentences_file_path}')
    SST_labels_file_path = os.path.join(SST_dir_path,'sentiment_labels.txt')
    if not os.path.isfile(SST_labels_file_path):
        print(f'NOT FOUND file: {SST_labels_file_path}')
    SST_dictionary_file_path = os.path.join(SST_dir_path,'dictionary.txt')
    if not os.path.isfile(SST_dictionary_file_path):
        print(f'NOT FOUND file: {SST_dictionary_file_path}')

    sentence_2_label_all, sentence_2_label_ternary = get_sentence_label_dict(SST_sentences_file_path, SST_labels_file_path, SST_dictionary_file_path)
    print('original ternary dataset size:', len(sentence_2_label_ternary))

    ZuCo_used_sentences = list(ZUCO_SENTIMENT_LABELS)

    filtered_ternary_dataset = {}
    filtered_pairs = []
    for key,value in sentence_2_label_ternary.items():
        add_instance = True
        for used_sent in ZuCo_used_sentences:
            if algorithims.trigram(used_sent, key) > 0.7:
                # print(f'Filter match: \n\t{used_sent}\n\t{key}')
                # print('###########################')
                filtered_pairs.append((used_sent, key))
                ZuCo_used_sentences.remove(used_sent)
                add_instance = False
                break
        if add_instance:
            filtered_ternary_dataset[key] = value
    
    print('filtered instance number:', len(filtered_pairs))
    print('filtered ternary dataset size:', len(filtered_ternary_dataset))
    print('unmatched remaining sentences:', ZuCo_used_sentences)
    print('unmatched remaining sentences length:', len(ZuCo_used_sentences))
    with open('temp.txt','w') as temp:
        for matched_pair in filtered_pairs:
            temp.write('#######\n')
            temp.write('\t'+matched_pair[0]+'\n')
            temp.write('\t'+matched_pair[1]+'\n')
            temp.write('\n')

    with open('./dataset/stanfordsentiment/ternary_dataset.json', 'w') as out:
        json.dump(filtered_ternary_dataset,out, indent = 4)
    print('write json to /dataset/stanfordsentiment/ternary_dataset.json')

if __name__ == '__main__':
    print('##############################')
    print('start generating stanfordSentimentTreebank ternary sentiment dataset...')
    SST_dir_path = '~/datasets/stanfordsentiment/stanfordSentimentTreebank'
    ZuCo_task1_csv_path = '~/datasets/ZuCo/task_materials/sentiment_labels_task1.csv'
    ZUCO_SENTIMENT_LABELS = json.load(open('~/datasets/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))

    get_SST_dataset(SST_dir_path, ZuCo_task1_csv_path, ZUCO_SENTIMENT_LABELS)