import os
import re
import time
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import random
from random import seed, randrange

import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu
from sklearn.model_selection import train_test_split

import pickle
import json
from tqdm import tqdm

from models.LSTM import Encoder, Decoder, Seq2Seq

root_dir = '/home/alfirsafauzulh@student.ub.ac.id/Firsa/Research/Chatbot/'
data_dir = root_dir + '/Datasets'
dailydialogs_root_dir = data_dir + '/dailydialog'
cornell_root_dir = data_dir + '/cornell_movie'

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
    
def respond_only(model, sentence, question, answer, device, max_length=50):
    # print(sentence)

    # sys.exit()

    # Load question tokenizer
    # spacy_en = spacy.load("en")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        sentence = normalize(sentence)
        sentence = remove_non_letter(sentence)
        sentence = remove_whitespace(sentence)

        tokens = [token.lower() for token in sentence.split(' ')]
    else:
        tokens = [token.lower() for token in sentence]

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, '<sos>')
    tokens.append('<eos>')

    # Go through each question token and convert to an index
    text_to_indices = []
    for token in tokens:
      if token in question.word2index.keys():
        text_to_indices.append(question.word2index[token])
      else:
        text_to_indices.append(question.word2index['<UNK>'])
    # text_to_indices = [question.word2index[token] for token in tokens]
    sentence_length = len(text_to_indices)
    text_to_indices = pad_sequences(text_to_indices, th+2)

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    sentence_length = torch.tensor([sentence_length])
    # Build encoder hidden, cell state

    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor, sentence_length)

    outputs = [answer.word2index["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == answer.word2index["<eos>"]:
            break

    answer_token = [answer.index2word[idx] for idx in outputs]

    # print('Question\t:', sentence)
    # print('Answer\t\t:', ' '.join(translated_sentence[1:-1]))
    
    return ' '.join(answer_token[1:-1])

def respond(sentence):
  answer = respond_only(model, sentence, question_tokenizer, answer_tokenizer, device, max_length=17)
  print('Me\t:', sentence)
  print('Bot\t:', answer)
  print()
    
    
np.random.seed(42)

th = 15
# df = pd.read_csv(dailydialogs_root_dir + f'/df_dailydialogs_max_{th}.csv')
df = pd.read_csv("/home/alfirsafauzulh@student.ub.ac.id/Firsa/Research/Chatbot" + f'/Code/Train/Datasets/dailydialog/df_dailydialogs_max_{th}.csv')


print(df.isnull().sum())
print()
df = df.dropna()
print()
print(df.isnull().sum())

tokenizer = Tokenizer(pd.concat([df['questions'], df['answers']], axis=0).values, min_freq=1)
question_tokenizer = tokenizer
answer_tokenizer = tokenizer

# question_tokenizer = Tokenizer(df['questions'].values, min_freq=1)
# answer_tokenizer = Tokenizer(df['answers'].values, min_freq=1)

print(len(question_tokenizer.vocab))
print(len(answer_tokenizer.vocab))

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

max_len = th+2

df['questions_preprocessed'] = df['questions'].map(lambda x: add_sos_eos(x))
df['answers_preprocessed'] = df['answers'].map(lambda x: add_sos_eos(x))

df['questions_preprocessed'] = df['questions_preprocessed'].map(lambda x: tokenize(x))
df['answers_preprocessed'] = df['answers_preprocessed'].map(lambda x: tokenize(x))

df['questions_preprocessed'] = df['questions_preprocessed'].map(lambda x: question_tokenizer.text_to_sequence(x))
df['answers_preprocessed'] = df['answers_preprocessed'].map(lambda x: answer_tokenizer.text_to_sequence(x))

df['questions_preprocessed'] = df['questions_preprocessed'].map(lambda x: pad_sequences(x, max_len))
df['answers_preprocessed'] = df['answers_preprocessed'].map(lambda x: pad_sequences(x, max_len))

df_train, df_test = train_test_split(df, test_size=.2, random_state=1111)

print(f"Jml Data Latih\t: {len(df_train)}\nJml Data Uji\t: {len(df_test)}")


input_tensor_train = df_train['questions_preprocessed'].values.tolist()
target_tensor_train = df_train['answers_preprocessed'].values.tolist()

input_tensor_test = df_test['questions_preprocessed'].values.tolist()
target_tensor_test = df_test['answers_preprocessed'].values.tolist()

train_data = MyData(input_tensor_train, target_tensor_train)
test_data = MyData(input_tensor_test, target_tensor_test)


np.random.seed(1111)
torch.manual_seed(1111)
torch.cuda.manual_seed(1111)

# Training hyperparams
num_epochs = 100
learning_rate = 0.0001
batch_size = 64

# Model hyperparams
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size_encoder = len(question_tokenizer.vocab)+4
input_size_decoder = len(answer_tokenizer.vocab)+4
output_size = len(answer_tokenizer.vocab)+4

encoder_embedding_size = 768
decoder_embedding_size = 768
hidden_size = 768
num_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5

train_dataset = DataLoader(train_data, batch_size = batch_size, drop_last=True, shuffle=False)
test_dataset = DataLoader(test_data, batch_size = batch_size, drop_last=True, shuffle=False)

# encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, 
#                       num_layers, enc_dropout, pretrained_word_embedding=True, embedding_matrix=embedding_matrix_q, freeze=False).to(device)

# decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, 
#                       output_size, num_layers, dec_dropout, pretrained_word_embedding=True, embedding_matrix=embedding_matrix_a, freeze=False).to(device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, 
                      num_layers, enc_dropout, pretrained_word_embedding=False, embedding_matrix=None, freeze=False).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, 
                      output_size, num_layers, dec_dropout, pretrained_word_embedding=False, embedding_matrix=None, freeze=False).to(device)

model = Seq2Seq(encoder_net, decoder_net, input_size_decoder).to(device)

pad_idx = answer_tokenizer.word2index["<PAD>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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


import gc
gc.collect()
torch.cuda.empty_cache()

train_losses = []
val_losses = []

for epoch in range(num_epochs):
  start = time.time()

  print(f"Epoch [{epoch+1}/{num_epochs}]")

  num_batch = 0
  val_num_batch = 0
  batch_loss = 0
  val_batch_loss = 0

  training_time = 0

  for (batch_idx, (X_train, y_train, input_len)) in enumerate(bar := tqdm(train_dataset)):
#   for (batch_idx, (X_train, y_train, input_len)) in enumerate(train_dataset):
    X, y, input_lengths = sort_within_batch(X_train, y_train, input_len)
    
    X = X.permute(1,0)
    y = y.permute(1,0)

    inp_data = X.to(device)
    target = y.to(device)
    # target shape = (target_length, batch_size))

    # print(inp_data.shape, target.shape)
    output = model(inp_data, target, input_lengths)
    # # output shape = (target_length, batch_size, output_dim)
    
    # print(output.shape, target.shape)

    output = output[1:].reshape(-1, output.shape[2])
    target = target[1:].reshape(-1)
    # membuat output shape menjadi (target_length*batch_size, output dim) dan target shape menjadi (target_length*batch_size) untuk dipassing ke loss function

    # print(output.shape, target.shape)

    optimizer.zero_grad()
    loss = loss_function(target, output)
    batch_loss += loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    
    bar.set_description(f'Train Seq2Seq Model '
                                         f'[train_loss={loss.detach():.4f}'
                                         )
    
#     if (batch_idx+1) % 32 == 0:
#       batch_avg_loss = batch_loss / (batch_idx+1)
#       print('\tEpoch {} Batch {} Loss {:.4f}'.format(epoch+1, batch_idx+1, batch_avg_loss))

    # writer.add_scalar("Training loss", loss, global_step=step)

    # step+=1
    num_batch+=1

  train_loss_ = batch_loss/num_batch
  train_losses.append(train_loss_)

  train_time = time.time() - start
  training_time += train_time

  print(f"Epoch {epoch+1} loss: {train_loss_}")
  print('Time taken for 1 epoch {} sec\n'.format(train_time))
    

with torch.no_grad():
  model.eval()
  test_batch_loss = 0
  test_num_batch = 0
  for (test_batch_idx, (X_test, y_test, input_len)) in enumerate(test_dataset):
    X, y, input_lengths = sort_within_batch(X_test, y_test, input_len)

    X = X.permute(1,0)
    y = y.permute(1,0)

    test_inp_data = X.to(device)
    test_target = y.to(device)

    output = model(test_inp_data, test_target, input_lengths)
      
    output = output[1:].reshape(-1, output.shape[2])
    test_target = test_target[1:].reshape(-1)

    test_loss = loss_function(test_target, output)
    test_batch_loss += test_loss
    test_num_batch+=1

  test_loss_ = test_batch_loss/test_num_batch

  print(f"test_loss: {test_loss_}")
    
test_questions = df_test['questions'].values
test_answers = df_test['answers'].values

preds = []
for x in test_questions:
  preds.append(respond_only(model, str(x), question_tokenizer, answer_tokenizer, device, max_length=th+2))


bleu_score_1 = 0
bleu_score_2 = 0
bleu_score_3 = 0
bleu_score_4 = 0
bleu_score_all = 0

num_of_rows_calculated = 0

for i, (question, real_answer) in enumerate(zip(test_questions, test_answers)):
  try:
    refs = [real_answer.split(' ')]
    hyp = preds[i].split(' ')

    bleu_score_1 += sentence_bleu(refs, hyp, weights=(1,0,0,0))
    bleu_score_2 += sentence_bleu(refs, hyp, weights=(0,1,0,0))
    bleu_score_3 += sentence_bleu(refs, hyp, weights=(0,0,1,0))
    bleu_score_4 += sentence_bleu(refs, hyp, weights=(0,0,0,1))
    bleu_score_all += sentence_bleu(refs, hyp, weights=(.25,.25,.25,.25))

    num_of_rows_calculated+=1
  except:
    print(f"EXCEPTION {(real_answer, preds[i])}")
    
print(f"Bleu Score 1-gram : {(bleu_score_1/num_of_rows_calculated)}")
print(f"Bleu Score 2-gram : {(bleu_score_2/num_of_rows_calculated)}")
print(f"Bleu Score 3-gram : {(bleu_score_3/num_of_rows_calculated)}")
print(f"Bleu Score 4-gram : {(bleu_score_4/num_of_rows_calculated)}")
print(f"Bleu Score all-gram : {(bleu_score_all/num_of_rows_calculated)}")