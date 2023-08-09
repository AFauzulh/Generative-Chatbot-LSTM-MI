import gc
import os
import re
import pickle
import json
import time
import numpy as np
import pandas as pd
import random
from random import seed, randrange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from sklearn.model_selection import train_test_split

from models.BiLSTM import Encoder, Decoder, Seq2Seq

from tqdm import tqdm
import argparse

import wandb

class Tokenizer():
  def __init__(self, data, min_freq=2, vocabs_npa=None, embs_npa=None):
    self.vocabs_npa = vocabs_npa
    self.embs_npa = embs_npa
    self.data = data
    self.min_freq = min_freq
    self.word2index = {}
    self.index2word = {}
    self.wordfreq = {}
    self.vocab = set()

    self.build()

  def build(self):
    for phrase in self.data:
      for word in phrase.split(' '):
        if word not in self.wordfreq.keys():
          self.wordfreq[word] = 1
        else:
          self.wordfreq[word]+=1

    for phrase in self.data:
      phrase_word = phrase.split(' ')
      phrase_word_update = []
      
      for data in phrase_word:
        if self.wordfreq[data] >= self.min_freq:
          phrase_word_update.append(data)

      self.vocab.update(phrase_word_update)

    self.vocab = sorted(self.vocab)

    self.word2index['<PAD>'] = 0
    self.word2index['<UNK>'] = 1
    self.word2index['<sos>'] = 2
    self.word2index['<eos>'] = 3
    
    for i, word in enumerate(self.vocab):
      self.word2index[word] = i+4

    for word, i in self.word2index.items():
      self.index2word[i] = word

  def text_to_sequence(self, text):
    sequences = []

    for word in text:
      try:
        sequences.append(self.word2index[word])
      except:
        sequences.append(self.word2index['<UNK>'])

    return sequences

  def sequence_to_text(self, sequence):
    texts = []

    for token in sequence:
      try:
        texts.append(self.index2word[token])
      except:
        texts.append(self.index2word[1])

    return texts

def pad_sequences(x, max_len):
  padded = np.zeros((max_len), dtype=np.int64)
  
  if len(x) > max_len:
    padded[:] = x[:max_len]

  else:
    padded[:len(x)] = x
    
  return padded

class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        # TODO: convert this into torch code is possible
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x,y,x_len
    
    def __len__(self):
        return len(self.data)    

def normalize(txt):
  txt = txt.lower()
  txt = re.sub(r"i'm", "i am", txt)
  txt = re.sub(r"he's", "he is", txt)
  txt = re.sub(r"she's", "she is", txt)
  txt = re.sub(r"that's", "that is", txt)
  txt = re.sub(r"what's", "what is", txt)
  txt = re.sub(r"where's", "where is", txt)
  txt = re.sub(r"\'ll", " will", txt)
  txt = re.sub(r"\'ve", " have", txt)
  txt = re.sub(r"\'re", " are", txt)
  txt = re.sub(r"\'d", " would", txt)
  txt = re.sub(r"won't", "will not", txt)
  txt = re.sub(r"can't", "can not", txt)
  txt = re.sub(r"a'ight", "alright", txt)
  txt = re.sub(r"n't", ' not', txt)
  return txt

def remove_non_letter(data):
  return re.sub(r'[^a-zA-Z]',' ', data)

def remove_whitespace(data):
  data = [x for x in data.split(' ') if x]
  return ' '.join(data)

def tokenize(text):
  text = str(text)
  return [token for token in text.split(' ')]

def add_sos_eos(text):
  return '<sos> ' + text + ' <eos>'

def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    #print(mask)
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    
    loss_ = criterion(pred, real) * mask 
    return torch.mean(loss_)

### sort batch function to be able to use with pad_packed_sequence
def sort_within_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths # transpose (batch x seq) to (seq x batch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--dataset_path', required=True, help='path to dataset (csv)')
    parser.add_argument('--manualSeed', type=int, default=77, help='for random seed setting')
    parser.add_argument('--model', required=True, help='select model')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=1024, help='the size of the LSTM hidden state')
    parser.add_argument('--embedding_dim', type=int, default=300, help='the size of the embedding dimension')
    parser.add_argument('--dropout', type=float, default=0, help='dropout ratio')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layer in RNN cell')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping value. default=5')
    parser.add_argument('--split_ratio', type=float, default=0.1,
                        help='assign ratio for split dataset')
    parser.add_argument('--max_length', type=int, default=15, help='maximum-sentence-length')
    parser.add_argument('--tokenizer_path', type=str, help='tokenizer path')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--wandb_log', action='store_true', help='using wandb')
    parser.add_argument('--transformer', action='store_true', help='using Transformer seq2seq model')
    parser.add_argument('--resume_train', action='store_true', help='resume training')
    
    opt = parser.parse_args()

    RANDOM_SEED = opt.manualSeed
    dataset_dir = opt.dataset_path
    
    if opt.exp_name is not None:
        os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    else:
        opt.exp_name = f"{opt.model}-{opt.dataset}"
        os.makedirs(f'./saved_models/{opt.model}-{opt.dataset}', exist_ok=True)
    
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)
    
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    
    df = pd.read_csv(opt.dataset_path)
    df = df.dropna()
    
    tokenizer = Tokenizer(pd.concat([df['questions'], df['answers']], axis=0).values, min_freq=1)
    question_tokenizer = tokenizer
    answer_tokenizer = tokenizer
    
    with open(f'./saved_models/{opt.exp_name}/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    max_len = opt.max_length+2
    
    df['questions_preprocessed'] = df['questions'].map(lambda x: add_sos_eos(x))
    df['answers_preprocessed'] = df['answers'].map(lambda x: add_sos_eos(x))

    df['questions_preprocessed'] = df['questions_preprocessed'].map(lambda x: tokenize(x))
    df['answers_preprocessed'] = df['answers_preprocessed'].map(lambda x: tokenize(x))

    df['questions_preprocessed'] = df['questions_preprocessed'].map(lambda x: question_tokenizer.text_to_sequence(x))
    df['answers_preprocessed'] = df['answers_preprocessed'].map(lambda x: answer_tokenizer.text_to_sequence(x))

    df['questions_preprocessed'] = df['questions_preprocessed'].map(lambda x: pad_sequences(x, max_len))
    df['answers_preprocessed'] = df['answers_preprocessed'].map(lambda x: pad_sequences(x, max_len))

    df_train, df_test = train_test_split(df, test_size=opt.split_ratio, random_state=RANDOM_SEED)
    df_train, df_val = train_test_split(df_train, test_size=.25, random_state=RANDOM_SEED)
    
    print(f"Train Data \t: {len(df_train)}")
    print(f"Val Data \t: {len(df_val)}")
    print(f"Test Data\t: {len(df_test)}\n")
    
    input_tensor_train = df_train['questions_preprocessed'].values.tolist()
    target_tensor_train = df_train['answers_preprocessed'].values.tolist()

    input_tensor_test = df_test['questions_preprocessed'].values.tolist()
    target_tensor_test = df_test['answers_preprocessed'].values.tolist()
    
    input_tensor_val = df_val['questions_preprocessed'].values.tolist()
    target_tensor_val = df_val['answers_preprocessed'].values.tolist()

    train_data = MyData(input_tensor_train, target_tensor_train)
    test_data = MyData(input_tensor_test, target_tensor_test)
    val_data = MyData(input_tensor_val, target_tensor_val)
    
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    num_epochs = opt.num_epochs
    learning_rate = opt.lr
    batch_size = opt.batch_size
    input_size_encoder = len(question_tokenizer.vocab)+4
    input_size_decoder = len(answer_tokenizer.vocab)+4
    output_size = len(answer_tokenizer.vocab)+4
    
    encoder_embedding_size = opt.embedding_dim
    decoder_embedding_size = opt.embedding_dim
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    enc_dropout = opt.dropout
    dec_dropout = opt.dropout
    
    train_dataset = DataLoader(train_data, batch_size = batch_size, drop_last=True, shuffle=True)
    test_dataset = DataLoader(test_data, batch_size = batch_size, drop_last=True, shuffle=True)
    val_dataset = DataLoader(val_data, batch_size = batch_size, drop_last=True, shuffle=True)
    
    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, 
                      num_layers, enc_dropout, pretrained_word_embedding=False, embedding_matrix=None, freeze=False).to(device)

    decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, 
                          output_size, num_layers, dec_dropout, pretrained_word_embedding=False, embedding_matrix=None, freeze=False).to(device)

    model = Seq2Seq(encoder_net, decoder_net, input_size_decoder).to(device)
    
    pad_idx = answer_tokenizer.word2index["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if opt.wandb_log:
        wandb.init(
            project="tesis-chatbot-siet-percobaan",
            entity='alfirsa-lab',
            name=f"{opt.exp_name}"
        )
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        num_batch = 0
        val_num_batch = 0
        batch_loss = 0
        val_batch_loss = 0
        training_time = 0
        
        for (batch_idx, (X_train, y_train, input_len)) in enumerate(bar := tqdm(train_dataset)):
            X, y, input_lengths = sort_within_batch(X_train, y_train, input_len)
            X = X.permute(1,0)
            y = y.permute(1,0)
            inp_data = X.to(device)
            target = y.to(device)
            
            output = model(inp_data, target, input_lengths)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            
            optimizer.zero_grad()
            loss = loss_function(target, output)
            # batch_loss += loss.detach()
            batch_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

#             bar.set_description(f'Train Seq2Seq Model '
#                                                  f'[train_loss={loss.detach():.4f}'
#                                                  )
            
            bar.set_description(f'Train Seq2Seq Model '
                                                 f'[train_loss={loss:.4f}'
                                                 )
            
            num_batch+=1
            train_loss_ = batch_loss/num_batch
        
        with torch.no_grad():
            for (batch_idx, (X_val, y_val, input_len)) in enumerate(val_dataset):
                X, y, input_lengths = sort_within_batch(X_val, y_val, input_len)
                X = X.permute(1,0)
                y = y.permute(1,0)
                
                inp_data = X.to(device)
                target = y.to(device)
                output = model(inp_data, target, input_lengths)
                
                output = output[1:].reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)
                
                val_loss = loss_function(target, output)
                val_batch_loss += val_loss
                
                val_num_batch+=1
                
        val_loss_ = val_batch_loss/val_num_batch
        scheduler.step(val_loss_)
        
        if val_loss_ < best_val_loss:
            torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_loss.pth')
            
        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/model.pth')
        
        if opt.wandb_log:
            wandb.log({"Train Loss": train_loss_, "Validation Loss": val_loss_, "Validation Perplexity": np.exp(val_loss_.cpu().detach().numpy())})
        
        print(
                f'Epochs: {epoch + 1} | Train Loss: {train_loss_:.3f} \
                | Val Loss: {val_loss_:.3f} | Val PPL: {np.exp(val_loss_.cpu().detach().numpy()):.3f} \n')
    
    if opt.wandb_log:
        wandb.finish()
        
    