import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from data import ZuCo_dataset, SST_tenary_dataset
from model_sentiment import FineTunePretrainedTwoStep
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

def train_model_ZuCo(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/text_sentiment_classifier/best/test.pt', checkpoint_path_last = './checkpoints/text_sentiment_classifier/last/test.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
      
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    best_acc = 0.0
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            total_accuracy = 0.0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_word_eeg_features, seq_lens, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders[phase]):
                
                # input_word_eeg_features = input_word_eeg_features.to(device).float()
                # input_masks = input_masks.to(device)
                # input_mask_invert = input_mask_invert.to(device)
                target_ids = target_ids.to(device)
                target_mask = target_mask.to(device)
                sentiment_labels = sentiment_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                output = model(input_ids = target_ids, attention_mask = target_mask, return_dict = True, labels = sentiment_labels)
                logits = output.logits
                loss = output.loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # calculate accuracy
                preds_cpu = logits.detach().cpu().numpy()
                label_cpu = sentiment_labels.cpu().numpy()

                total_accuracy += flat_accuracy(preds_cpu, label_cpu)

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
            if phase == 'dev' and (epoch_acc > best_acc):
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best val acc: {:4f}'.format(best_acc))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')
    
    # write to log
    with open(output_log_file_name, 'w') as outlog:
        outlog.write(f'best val loss: {best_loss}\n')
        outlog.write('Best val acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_model_SST(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/text_sentiment_classifier/best/test.pt', checkpoint_path_last = './checkpoints/text_sentiment_classifier/last/test.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
      
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    best_acc = 0.0
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            total_accuracy = 0.0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_ids,input_masks,sentiment_labels in tqdm(dataloaders[phase]):
                
                input_ids = input_ids.to(device)
                input_masks = input_masks.to(device)
                sentiment_labels = sentiment_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                output = model(input_ids = input_ids, attention_mask = input_masks, return_dict = True, labels = sentiment_labels)
                logits = output.logits
                loss = output.loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # calculate accuracy
                preds_cpu = logits.detach().cpu().numpy()
                label_cpu = sentiment_labels.cpu().numpy()

                total_accuracy += flat_accuracy(preds_cpu, label_cpu)

                # statistics
                running_loss += loss.item() * input_ids.size()[0] # batch loss
                # print('[DEBUG]loss:',loss.item())
                # print('#################################')
                

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = total_accuracy / len(dataloaders[phase])
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} Acc: {:.4f}'.format(phase, epoch_acc))

            # deep copy the model
            if phase == 'dev' and (epoch_acc > best_acc):
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best val acc: {:4f}'.format(best_acc))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    args = get_config('train_sentiment_textbased')

    ''' config param'''

    num_epoch = args['num_epoch']
    # lr = 1e-3 # Bert, RoBerta
    # lr = 1e-4 # Bart
    lr = args['learning_rate']

    dataset_name = args['dataset_name'] # zero-shot setting: using external dataset from stanford sentiment treebank, pass in 'SST'; or pass in 'ZuCo' to train on ZuCo's text-sentiment pairs

    dataset_setting = 'unique_sent'

    batch_size = args['batch_size']
    
    # model_name = 'pretrain_Bert'
    # model_name = 'pretrain_RoBerta'
    # model_name = 'pretrain_Bart'
    model_name = args['model_name']
    print(f'[INFO]model name: {model_name}')

    save_path = args['save_path'] 

    if dataset_name == 'ZuCo':
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
        save_name = f'Textbased_ZuCo_{model_name}_b{batch_size}_{num_epoch}_{lr}_{dataset_setting}_{eeg_type_choice}'
    elif dataset_name == 'SST':
        save_name = f'Textbased_StanfordSentitmentTreeband_{model_name}_b{batch_size}_{num_epoch}_{lr}'

    output_checkpoint_name_best = save_path + f'/best/{save_name}.pt' 
    output_checkpoint_name_last = save_path + f'/last/{save_name}.pt' 


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


    ''' load pickle '''
    if dataset_name == 'ZuCo':
        whole_dataset_dict = []
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dict.append(pickle.load(handle))
    
    '''tokenizer'''
    if model_name == 'pretrain_Bert':
        print('[INFO]pretrained checkpoint: bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model_name == 'pretrain_RoBerta':
        print('[INFO]pretrained checkpoint: roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif model_name == 'pretrain_Bart':
        print('[INFO]pretrained checkpoint: bart-large')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    ''' set up dataloader '''
    if dataset_name == 'ZuCo':
        # train dataset
        train_set = ZuCo_dataset(whole_dataset_dict, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
        # dev dataset
        dev_set = ZuCo_dataset(whole_dataset_dict, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    
    elif dataset_name == 'SST':
        SST_SENTIMENT_LABELS = json.load(open('./dataset/stanfordsentiment/ternary_dataset.json'))

        SST_dataset = SST_tenary_dataset(SST_SENTIMENT_LABELS, tokenizer)  
        
        train_size = int(0.9 * len(SST_dataset))
        val_size = len(SST_dataset) - train_size

        train_set, dev_set = random_split(SST_dataset, [train_size, val_size])
        print('{:>5,} training samples'.format(len(train_set)))
        print('{:>5,} validation samples'.format(len(dev_set)))


    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}

    ''' set up model '''
    if model_name == 'pretrain_Bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=3)
    elif model_name == 'pretrain_RoBerta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    elif model_name == 'pretrain_Bart':
        model = BartForSequenceClassification.from_pretrained('facebook/bart-large', num_labels = 3)
    
    model.to(device)
    

    """save config"""
    with open(f'./config/text_sentiment_classifier/{save_name}.json', 'w') as out_config:
        json.dump(args, out_config, indent = 4)


    ''' training loop '''
    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    ''' set up optimizer and scheduler'''
    optimizer_step1 = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=10, gamma=0.1)

    # TODO: rethink about the loss function
    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    # return best loss model from step1 training
    print(f'=== start training {dataset_name} ... ===')
    if dataset_name == 'ZuCo':
        model = train_model_ZuCo(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epoch, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)
    elif dataset_name == 'SST':
        model = train_model_SST(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epoch, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)
        