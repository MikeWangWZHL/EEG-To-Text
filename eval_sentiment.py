import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pack_padded_sequence 
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from data import ZuCo_dataset
from model_sentiment import BaselineMLPSentence, BaselineLSTM, FineTunePretrainedTwoStep, ZeroShotSentimentDiscovery, JointBrainTranslatorSentimentClassifier
from model_decoding import BrainTranslator, BrainTranslatorNaive
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from config import get_config

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    # preds: numpy array: N * 3 
    # labels: numpy array: N 
    pred_flat = np.argmax(preds, axis=1).flatten()  
    
    labels_flat = labels.flatten()
    
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_accuracy_top_k(preds, labels,k):
    topk_preds = []
    for pred in preds:
        topk = pred.argsort()[-k:][::-1]
        topk_preds.append(list(topk))
    # print(topk_preds)
    topk_preds = list(topk_preds)
    right_count = 0
    # print(len(labels))
    for i in range(len(labels)):
        l = labels[i][0]
        if l in topk_preds[i]:
            right_count+=1
    return right_count/len(labels)

def eval_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')):

    def logits2PredString(logits, tokenizer):
        probs = logits[0].softmax(dim = 1)
        # print('probs size:', probs.size())
        values, predictions = probs.topk(1)
        # print('predictions before squeeze:',predictions.size())
        predictions = torch.squeeze(predictions)
        predict_string = tokenizer.decode(predictions)
        return predict_string
    
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
      
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    best_acc = 0.0
    
    total_pred_labels = np.array([])
    total_true_labels = np.array([])

    for epoch in range(1):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['test']:
            total_accuracy = 0.0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_word_eeg_features, seq_lens, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in dataloaders[phase]:
                
                input_word_eeg_features = input_word_eeg_features.to(device).float()
                input_masks = input_masks.to(device)
                input_mask_invert = input_mask_invert.to(device)
 
                sent_level_EEG = sent_level_EEG.to(device)
                sentiment_labels = sentiment_labels.to(device)

                target_ids = target_ids.to(device)
                target_mask = target_mask.to(device)

                ## forward ###################
                if isinstance(model, BaselineMLPSentence):
                    logits = model(sent_level_EEG) # before softmax
                    # calculate loss
                    loss = criterion(logits, sentiment_labels)

                elif isinstance(model, BaselineLSTM):
                    x_packed = pack_padded_sequence(input_word_eeg_features, seq_lens, batch_first=True, enforce_sorted=False)
                    logits = model(x_packed)
                    # calculate loss
                    loss = criterion(logits, sentiment_labels)

                elif isinstance(model, BertForSequenceClassification) or isinstance(model, RobertaForSequenceClassification) or isinstance(model, BartForSequenceClassification):
                    output = model(input_ids = target_ids, attention_mask = target_mask, return_dict = True, labels = sentiment_labels)
                    logits = output.logits
                    loss = output.loss
                
                elif isinstance(model, FineTunePretrainedTwoStep):
                    output = model(input_word_eeg_features, input_masks, input_mask_invert, sentiment_labels)
                    logits = output.logits
                    loss = output.loss

                elif isinstance(model, ZeroShotSentimentDiscovery):    
                    print()
                    print('target string:',tokenizer.decode(target_ids[0]).replace('<pad>','').split('</s>')[0]) 

                    """replace padding ids in target_ids with -100"""
                    target_ids[target_ids == tokenizer.pad_token_id] = -100 

                    output = model(input_word_eeg_features, input_masks, input_mask_invert, target_ids, sentiment_labels)
                    logits = output.logits
                    loss = output.loss
                
                elif isinstance(model, JointBrainTranslatorSentimentClassifier):

                    print()
                    print('target string:',tokenizer.decode(target_ids[0]).replace('<pad>','').split('</s>')[0]) 

                    """replace padding ids in target_ids with -100"""
                    target_ids[target_ids == tokenizer.pad_token_id] = -100 

                    LM_output, classification_output = model(input_word_eeg_features, input_masks, input_mask_invert, target_ids, sentiment_labels)
                    LM_logits = LM_output.logits
                    print('pred string:', logits2PredString(LM_logits, tokenizer).split('</s></s>')[0].replace('<s>',''))
                    classification_loss = classification_output['loss']
                    logits = classification_output['logits']
                    loss = classification_loss 
                ###############################

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # calculate accuracy
                preds_cpu = logits.detach().cpu().numpy()
                label_cpu = sentiment_labels.cpu().numpy()

                total_accuracy += flat_accuracy(preds_cpu, label_cpu)
                
                # add to total pred and label array, for cal F1, precision, recall
                pred_flat = np.argmax(preds_cpu, axis=1).flatten()
                labels_flat = label_cpu.flatten()

                total_pred_labels = np.concatenate((total_pred_labels,pred_flat))
                total_true_labels = np.concatenate((total_true_labels,labels_flat))
                

                # statistics
                running_loss += loss.item() * sent_level_EEG.size()[0] # batch loss
                # print('[DEBUG]loss:',loss.item())
                # print('#################################')
                

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = total_accuracy / len(dataloaders[phase])
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} Acc: {:.4f}'.format(phase, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test loss: {:4f}'.format(best_loss))
    print('Best test acc: {:4f}'.format(best_acc))
    print()
    print('test sample num:', len(total_pred_labels))
    print('total preds:',total_pred_labels)
    print('total truth:',total_true_labels)
    print('sklearn macro: precision, recall, F1:')
    print(precision_recall_fscore_support(total_true_labels, total_pred_labels, average='macro'))
    print()
    print('sklearn micro: precision, recall, F1:')
    print(precision_recall_fscore_support(total_true_labels, total_pred_labels, average='micro'))
    print()
    print('sklearn accuracy:')
    print(accuracy_score(total_true_labels,total_pred_labels))
    print()



if __name__ == '__main__':
    args = get_config('eval_sentiment')

    ''' config param'''
    num_epochs = 1

    dataset_setting = 'unique_sent'
    
    '''model name'''
    # model_name = 'BaselineMLP'
    # model_name = 'BaselineLSTM'
    # model_name = 'NaiveFinetuneBert'
    # model_name = 'FinetunedBertOnText'
    # model_name = 'FinetunedRoBertaOnText'
    # model_name = 'FinetunedBartOnText'
    # model_name = 'ZeroShotSentimentDiscovery'
    model_name = args['model_name']

    print(f'[INFO] eval {model_name}')
    if model_name == 'ZeroShotSentimentDiscovery':
        '''load decoder and classifier config'''
        config_decoder = json.load(open(args['decoder_config_path']))
        config_classifier = json.load(open(args['classifier_config_path']))
        '''choose generator'''
        # decoder_name = 'BrainTranslator'
        # decoder_name = 'BrainTranslatorNaive'
        decoder_name = config_decoder['model_name']
        decoder_checkpoint = args['decoder_checkpoint_path']
        print(f'[INFO] using decoder: {decoder_name}')

        '''choose classifier'''
        # pretrain_Bert, pretrain_RoBerta, pretrain_Bart
        classifier_name = config_classifier['model_name']
        classifier_checkpoint = args['classifier_checkpoint_path']
        print(f'[INFO] using classifier: {classifier_name}')
    else:
        checkpoint_path = args['checkpoint_path']
        print('[INFO] loading baseline:', checkpoint_path)

    batch_size = 1


    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')


    
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


    ''' load pickle'''
    whole_dataset_dict = []
    dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle' 
    with open(dataset_path_task1, 'rb') as handle:
        whole_dataset_dict.append(pickle.load(handle))
    
    '''set up tokenizer'''
    if model_name in ['BaselineMLP','BaselineLSTM', 'NaiveFinetuneBert', 'FinetunedBertOnText']:
        print('[INFO]using Bert tokenizer')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model_name == 'FinetunedBartOnText':
        print('[INFO]using Bart tokenizer')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif model_name == 'FinetunedRoBertaOnText':
        print('[INFO]using RoBerta tokenizer')
        tokenizer =  RobertaTokenizer.from_pretrained('roberta-base')
    elif model_name == 'ZeroShotSentimentDiscovery':
        decoder_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large') # Bart
        tokenizer = decoder_tokenizer
        if classifier_name == 'pretrain_Bert':
            sentiment_tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # Bert
        elif classifier_name == 'pretrain_Bart':
            sentiment_tokenizer = decoder_tokenizer
        elif classifier_name == 'pretrain_RoBerta':
            sentiment_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    ''' set up model '''
    if model_name == 'BaselineMLP':
        print('[INFO]Model: BaselineMLP')
        model = BaselineMLPSentence(input_dim = 840, hidden_dim = 128, output_dim = 3)
    elif model_name == 'BaselineLSTM':
        print('[INFO]Model: BaselineLSTM')
        # model = BaselineLSTM(input_dim = 840, hidden_dim = 256, output_dim = 3, num_layers = 1)
        model = BaselineLSTM(input_dim = 840, hidden_dim = 256, output_dim = 3, num_layers = 4)
    elif model_name == 'FinetunedBertOnText':
        print('[INFO]Model: FinetunedBertOnText')
        model = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=3)
    elif model_name == 'FinetunedRoBertaOnText':
        print('[INFO]Model: FinetunedRoBertaOnText')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    elif model_name == 'FinetunedBartOnText':
        print('[INFO]Model: FinetunedBartOnText')
        model = BartForSequenceClassification.from_pretrained('facebook/bart-large', num_labels=3)
    elif model_name == 'ZeroShotSentimentDiscovery':
        print(f'[INFO]Model: ZeroShotSentimentDiscovery, using classifer:{classifier_name}, using generator: {decoder_name}')
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        if decoder_name == 'BrainTranslator':
            decoder = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
        elif decoder_name == 'BrainTranslatorNaive':
            decoder = BrainTranslatorNaive(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
        decoder.load_state_dict(torch.load(decoder_checkpoint))
        
        if classifier_name == 'pretrain_Bert':
            classifier = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=3)
        elif classifier_name == 'pretrain_Bart':
            classifier = BartForSequenceClassification.from_pretrained('facebook/bart-large', num_labels=3)
        elif classifier_name == 'pretrain_RoBerta':
            classifier = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

        classifier.load_state_dict(torch.load(classifier_checkpoint))

        model = ZeroShotSentimentDiscovery(decoder, classifier, decoder_tokenizer, sentiment_tokenizer, device = device)
        model.to(device)

    if model_name != 'ZeroShotSentimentDiscovery':
        # load model and send to device
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)

    ''' set up dataloader '''
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = 'unique_sent')

    dataset_sizes = {'test': len(test_set)}
    # print('[INFO]train_set size: ', len(train_set))
    print('[INFO]test_set size: ', len(test_set))
    
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'test':test_dataloader}
    
    ''' set up optimizer and scheduler'''
    optimizer_step1 = None
    exp_lr_scheduler_step1 = None

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print('=== start training ... ===')
    # return best loss model from step1 training
    model = eval_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs, tokenizer = tokenizer)
