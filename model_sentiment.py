import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BertForSequenceClassification
import math
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""MLP baseline using sentence level eeg"""
# using sent level EEG, MLP baseline for sentiment
class BaselineMLPSentence(nn.Module):
    def __init__(self, input_dim = 840, hidden_dim = 128, output_dim = 3):
        super(BaselineMLPSentence, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim) # positive, negative, neutral  
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


"""bidirectional LSTM baseline using word level eeg"""
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim = 840, hidden_dim = 256, output_dim = 3, num_layers = 1):
        super(BaselineLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = 1, batch_first = True, bidirectional = True)

        self.hidden2sentiment = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x_packed):
        # input: (N,seq_len,input_dim)
        # print(x_packed.data.size())
        lstm_out, _ = self.lstm(x_packed)
        last_hidden_state = pad_packed_sequence(lstm_out, batch_first = True)[0][:,-1,:]
        # print(last_hidden_state.size())
        out = self.hidden2sentiment(last_hidden_state)
        return out

""" Bert Baseline: Finetuning from a pretrained language model Bert"""
class NaiveFineTunePretrainedBert(nn.Module):
    def __init__(self, input_dim = 840, hidden_dim = 768, output_dim = 3, pretrained_checkpoint = None):
        super(NaiveFineTunePretrainedBert, self).__init__()
        # mapping hidden states dimensioin
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.pretrained_Bert = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=3)
        
        if pretrained_checkpoint is not None:
            self.pretrained_Bert.load_state_dict(torch.load(pretrained_checkpoint))

    def forward(self, input_embeddings_batch, input_masks_batch, labels):
        embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained_Bert(inputs_embeds = embedding, attention_mask = input_masks_batch, labels = labels, return_dict = True)
        return out

""" Finetuning from a pretrained language model BART, two step training"""
class FineTunePretrainedTwoStep(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, d_model = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(FineTunePretrainedTwoStep, self).__init__()
        
        self.pretrained_layers = pretrained_layers
        # additional transformer encoder, following BART paper about 
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
        # NOTE: add positional embedding?
        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, d_model)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, labels):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        """labels: sentitment labels 0,1,2"""
        
        # NOTE: add positional embedding?
        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch) 

        # use src_key_padding_masks
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask = input_masks_invert) 
        # encoded_embedding = self.additional_encoder(input_embeddings_batch) 
        
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        out = self.pretrained_layers(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, return_dict = True, labels = labels)                    
        
        return out

""" Zero-shot sentiment discovery using a finetuned generation model and a sentiment model pretrained on text """
class ZeroShotSentimentDiscovery(nn.Module):
    def __init__(self, brain2text_translator, sentiment_classifier, translation_tokenizer, sentiment_tokenizer, device = 'cpu'):
        # only for inference
        super(ZeroShotSentimentDiscovery, self).__init__()
        
        self.brain2text_translator = brain2text_translator
        self.sentiment_classifier = sentiment_classifier
        self.translation_tokenizer = translation_tokenizer
        self.sentiment_tokenizer = sentiment_tokenizer
        self.device = device
    

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, sentiment_labels):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        """labels: sentitment labels 0,1,2"""
        
        def logits2PredString(logits):
            probs = logits[0].softmax(dim = 1)
            # print('probs size:', probs.size())
            values, predictions = probs.topk(1)
            # print('predictions before squeeze:',predictions.size())
            predictions = torch.squeeze(predictions)
            predict_string = self.translation_tokenizer.decode(predictions)
            return predict_string

        # only works on batch is one
        assert input_embeddings_batch.size()[0] == 1

        seq2seqLMoutput = self.brain2text_translator(input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted)
        predict_string = logits2PredString(seq2seqLMoutput.logits)
        predict_string = predict_string.split('</s></s>')[0]
        predict_string = predict_string.replace('<s>','')
        print('predict string:', predict_string)
        re_tokenized = self.sentiment_tokenizer(predict_string, return_tensors='pt', return_attention_mask = True)
        input_ids = re_tokenized['input_ids'].to(self.device) # batch = 1
        attn_mask = re_tokenized['attention_mask'].to(self.device) # batch = 1

        out = self.sentiment_classifier(input_ids = input_ids, attention_mask = attn_mask, return_dict = True, labels = sentiment_labels)

        return out


""" Miscellaneous: jointly learn generation and classification (not working well) """
class BartClassificationHead(nn.Module):
    # from transformers: https://huggingface.co/transformers/_modules/transformers/models/bart/modeling_bart.html
    """Head for sentence-level classification tasks."""
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class JointBrainTranslatorSentimentClassifier(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, d_model = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048, num_labels = 3):
        super(JointBrainTranslatorSentimentClassifier, self).__init__()
        
        self.pretrained_generator = pretrained_layers
        # additional transformer encoder, following BART paper about 
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        self.fc1 = nn.Linear(in_feature, d_model)
        self.num_labels = num_labels

        self.pooler = Pooler(d_model)
        self.classifier = BartClassificationHead(input_dim = d_model, inner_dim = d_model, num_classes = num_labels, pooler_dropout = pretrained_layers.config.classifier_dropout)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, sentiment_labels):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        
        # NOTE: add positional embedding?
        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch) 

        # use src_key_padding_masks
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask = input_masks_invert) 
        
        # encoded_embedding = self.additional_encoder(input_embeddings_batch) 
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        LMoutput = self.pretrained_generator(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, return_dict = True, labels = target_ids_batch_converted, output_hidden_states = True)                    
        hidden_states = LMoutput.decoder_hidden_states # N, seq_len, hidden_dim
        # print('hidden states len:', len(hidden_states))
        last_hidden_states = hidden_states[-1]
        # print('last hidden states size:', last_hidden_states.size())
        sentence_representation = self.pooler(last_hidden_states)
 
        classification_logits = self.classifier(sentence_representation) 
        loss_fct = nn.CrossEntropyLoss()
        classification_loss = loss_fct(classification_logits.view(-1, self.num_labels), sentiment_labels.view(-1))
        classification_output = {'loss':classification_loss,'logits':classification_logits}
        # print('successful one forward!!!!')
        return LMoutput, classification_output


""" helper modules """
# modified from BertPooler
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('[DEBUG] input size:', x.size())
        # print('[DEBUG] positional embedding size:', self.pe.size())
        x = x + self.pe[:x.size(0), :]
        # print('[DEBUG] output x with pe size:', x.size())
        return self.dropout(x)

