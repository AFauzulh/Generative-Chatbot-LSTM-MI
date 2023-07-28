import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

RANDOM_SEED = 61
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, pretrained_word_embedding=False, embedding_matrix=None, freeze=False):
    super(Encoder, self).__init__()
    self.dropout = nn.Dropout(p)
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.device = device

    if pretrained_word_embedding:
      self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix).float(), freeze=freeze)
    
    self.embedding = nn.Embedding(input_size, embedding_size)

    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

  
  def forward(self, x, lens):
    # x shape: (seq_length, N), N ==> batch size

    embedding = self.dropout(self.embedding(x))
    # embedding shape: (ssq_length, N, embedding_size)

    embedding = pack_padded_sequence(embedding, lens)

    outputs, (hidden, cell) = self.rnn(embedding)

    outputs, _ = pad_packed_sequence(outputs)

    return hidden, cell

class Decoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, output_size, 
               num_layers, p, pretrained_word_embedding=False, embedding_matrix=None, freeze=False):
    
    super(Decoder, self).__init__()
    self.dropout = nn.Dropout(p)
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    if pretrained_word_embedding:
      self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix).float(), freeze=freeze)
    
    self.embedding = nn.Embedding(input_size, embedding_size)

    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    self.fc = nn.Linear(hidden_size, output_size)


  def forward(self, x, hidden, cell):
    # x shape = (N) tapi kita butuh (1, N) karena decoder hanya predict 1 kata tiap predict
    x = x.unsqueeze(0)

    embedding = self.dropout(self.embedding(x))
    # embedding shape = (1, N, embedding_size)

    outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
    # outputs shape = (1, N, hidden_size)

    predictions = self.fc(outputs)
    # predictions shape = (1, N, length_vocab)

    predictions = predictions.squeeze(0)
    # predictions shape = (N, length_vocab) untuk dipassing ke loss function

    return predictions, hidden, cell

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, vocab_len):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.vocab_len = vocab_len

  def forward(self, source, target, input_len, teacher_force_ratio=0):
    # source and target shape = (target_len, N)
    batch_size = source.shape[1]
    target_len = target.shape[0]
    target_vocabulary_size = self.vocab_len
    # target_vocabulary_size = len(answer_tokenizer.vocab)

    outputs = torch.zeros(target_len, batch_size, target_vocabulary_size).to(device)

    hidden, cell = self.encoder(source, input_len)
    # ambil start token
    x = target[0]

    for t in range(1, target_len):
      output, hidden, cell = self.decoder(x, hidden, cell)

      outputs[t] = output
      # output shape = (N, answer_vocab_size)

      best_guess = output.argmax(1)

      x = target[t] if random.random() < teacher_force_ratio else best_guess

    return outputs