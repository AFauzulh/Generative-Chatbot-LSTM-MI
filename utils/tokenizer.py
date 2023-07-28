import torch

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocess import normalize, remove_non_letter, remove_whitespace
RANDOM_SEED = 61

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.target_length = [ np.sum(1 - np.equal(trg, 0)) for trg in y]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        y_len = self.target_length[index]
        return x,y,x_len,y_len
    
    def __len__(self):
        return len(self.data)
    
# class MyData(Dataset):
#     def __init__(self, X, y):
#         self.data = X
#         self.target = y
#         # TODO: convert this into torch code is possible
#         self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.target[index]
#         x_len = self.length[index]
#         return x,y,x_len
    
#     def __len__(self):
#         return len(self.data)
    
def respond_only_lstm_attn(model, sentence, question, answer, device, max_length):
    if type(sentence) == str:
        sentence = normalize(sentence)
        sentence = remove_non_letter(sentence)
        sentence = remove_whitespace(sentence)

        tokens = [token.lower() for token in sentence.split(' ')]
    else:
        tokens = [token.lower() for token in sentence]

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
    text_to_indices = pad_sequences(text_to_indices, max_length)

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    sentence_length = torch.tensor([sentence_length])
    # Build encoder hidden, cell state

    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(sentence_tensor, sentence_length)

    outputs = [answer.word2index["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell, _ = model.decoder(previous_word, encoder_states, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == answer.word2index["<eos>"]:
            break

    answer_token = [answer.index2word[idx] for idx in outputs]

    return ' '.join(answer_token[1:-1])

def respond_only_lstm_no_attn(model, sentence, question, answer, device, max_length):
    if type(sentence) == str:
        sentence = normalize(sentence)
        sentence = remove_non_letter(sentence)
        sentence = remove_whitespace(sentence)

        tokens = [token.lower() for token in sentence.split(' ')]
    else:
        tokens = [token.lower() for token in sentence]

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
    text_to_indices = pad_sequences(text_to_indices, max_length)

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
            output, hidden, cell= model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == answer.word2index["<eos>"]:
            break

    answer_token = [answer.index2word[idx] for idx in outputs]

    return ' '.join(answer_token[1:-1])

def respond_only_gru_no_attn(model, sentence, question, answer, device, max_length=50):
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
    text_to_indices = pad_sequences(text_to_indices, max_length)

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    sentence_length = torch.tensor([sentence_length])
    # Build encoder hidden, cell state

    with torch.no_grad():
        hidden = model.encoder(sentence_tensor, sentence_length)

    outputs = [answer.word2index["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(previous_word, hidden)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == answer.word2index["<eos>"]:
            break

    answer_token = [answer.index2word[idx] for idx in outputs]

    # print('Question\t:', sentence)
    # print('Answer\t\t:', ' '.join(translated_sentence[1:-1]))
    
    return ' '.join(answer_token[1:-1])

def respond_only_gru_attn(model, sentence, question, answer, device, max_length=50):
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
    text_to_indices = pad_sequences(text_to_indices, max_length)

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    sentence_length = torch.tensor([sentence_length])
    # Build encoder hidden, cell state

    with torch.no_grad():
        encoder_states, hidden = model.encoder(sentence_tensor, sentence_length)

    outputs = [answer.word2index["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, _ = model.decoder(previous_word, encoder_states, hidden)
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