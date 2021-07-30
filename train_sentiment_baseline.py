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

from transformers import BertTokenizer, BertLMHeadModel, BertConfig
from data import ZuCo_dataset
from model_sentiment import BaselineMLPSentence, BaselineLSTM, NaiveFineTunePretrainedBert
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

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/eeg_sentiment/best/test.pt', checkpoint_path_last = './checkpoints/eeg_sentiment/last/test.pt'):
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
                
                input_word_eeg_features = input_word_eeg_features.to(device).float()
                sent_level_EEG = sent_level_EEG.to(device)
                input_masks = input_masks.to(device)
                sentiment_labels = sentiment_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                if isinstance(model, BaselineMLPSentence):
                    # forward
                    logits = model(sent_level_EEG) # before softmax
                    # calculate loss
                    loss = criterion(logits, sentiment_labels)
                
                elif isinstance(model, BaselineLSTM):
                    x_packed = pack_padded_sequence(input_word_eeg_features, seq_lens, batch_first=True, enforce_sorted=False)
                    logits = model(x_packed)
                    # calculate loss
                    loss = criterion(logits, sentiment_labels)
                
                elif isinstance(model, NaiveFineTunePretrainedBert):
                    output = model(input_word_eeg_features, input_masks, sentiment_labels)
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


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    args = get_config('train_sentiment_baseline')
    
    ''' config param'''
    num_epochs = args['num_epoch']
    step_lr = args['learning_rate']

    '''dataset division'''
    dataset_setting = 'unique_sent'

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
    
    '''model name'''
    # model_name = 'BaselineMLP'
    # model_name = 'BaselineLSTM'
    # model_name = 'NaiveFinetuneBert'
    model_name = args['model_name']

    batch_size = 32
    save_path = args['save_path']
    save_name = f'{model_name}_{step_lr}_b{batch_size}_{dataset_setting}_{eeg_type_choice}'

    if model_name == 'BaselineLSTM':
        num_layers = 4
        save_name = f'{model_name}_numLayers-{num_layers}_{step_lr}_b{batch_size}_{dataset_setting}_{eeg_type_choice}'

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


    ''' load pickle'''
    whole_dataset_dict = []
    dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle' 
    with open(dataset_path_task1, 'rb') as handle:
        whole_dataset_dict.append(pickle.load(handle))
    
    '''set up tokenizer'''
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    ''' set up dataloader '''
    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dict, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dict, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)

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
    if model_name == 'BaselineMLP':
        print('[INFO]Model: BaselineMLP')
        model = BaselineMLPSentence(input_dim = 105*len(bands_choice), hidden_dim = 128, output_dim = 3)
    elif model_name == 'BaselineLSTM':
        print('[INFO]Model: BaselineLSTM')
        model = BaselineLSTM(input_dim = 105*len(bands_choice), hidden_dim = 256, output_dim = 3, num_layers = num_layers)
    elif model_name == 'NaiveFinetuneBert':
        print('[INFO]Model: NaiveFinetuneBert')
        model = NaiveFineTunePretrainedBert(input_dim = 105*len(bands_choice), hidden_dim = 768, output_dim = 3)
    
    model.to(device)
    

    """save config"""
    with open(f'./config/eeg_sentiment/{save_name}.json', 'w') as out_config:
        json.dump(args, out_config, indent = 4)
    
    
    ''' training loop '''

    ''' set up optimizer and scheduler'''
    optimizer_step1 = optim.SGD(model.parameters(), lr=step_lr, momentum=0.9)
    exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.5)

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print('=== start training ... ===')
    # return best loss model from step1 training
    model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)
