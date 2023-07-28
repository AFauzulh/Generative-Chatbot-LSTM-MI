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
    
    encoder_states, (hidden, cell) = self.rnn(embedding)
    encoder_states, _ = pad_packed_sequence(encoder_states)

    return encoder_states, hidden, cell

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

    self.rnn = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, dropout=p)
    self.fc = nn.Linear(hidden_size, output_size)

    self.W1 = nn.Linear(self.hidden_size, self.hidden_size)
    self.W2 = nn.Linear(self.hidden_size, self.hidden_size)
    self.V = nn.Linear(self.hidden_size, 1)

  def forward(self, x, encoder_states, hidden, cell):
    # x shape = (N) tapi kita butuh (1, N) karena decoder hanya predict 1 kata tiap predict
    x = x.unsqueeze(0)

    encoder_states = encoder_states.permute(1, 0, 2)
    hidden_with_time_axis = hidden.permute(1, 0, 2)

    attention_score = self.W1(encoder_states) + self.W2(hidden_with_time_axis)
    attention_score = torch.tanh(attention_score)

    attention_weights = self.V(attention_score)
    attention_weights = torch.softmax(attention_weights, dim=1)

    context_vector = attention_weights * encoder_states
    context_vector = torch.sum(context_vector, dim=1)

    embedding = self.dropout(self.embedding(x))
    # embedding shape = (1, N, embedding_size)

    embedding = torch.cat((context_vector.unsqueeze(1).permute(1,0,2), embedding), dim=-1)

    outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
    # outputs shape = (1, N, hidden_size)

    predictions = self.fc(outputs)
    # predictions shape = (1, N, length_vocab)

    predictions = predictions.squeeze(0)
    # predictions shape = (N, length_vocab) untuk dipassing ke loss function

    return predictions, hidden, cell, attention_weights

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, vocab_len):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.vocab_len = vocab_len

  def forward(self, source, target, input_len, teacher_force_ratio=.5):
    # source and target shape = (target_len, N)
    batch_size = source.shape[1]
    target_len = target.shape[0]
    target_vocabulary_size = self.vocab_len
    # target_vocabulary_size = len(answer_tokenizer.vocab)

    outputs = torch.zeros(target_len, batch_size, target_vocabulary_size).to(device)

    encoder_states, hidden, cell = self.encoder(source, input_len)
    # ambil start token
    x = target[0]

    for t in range(1, target_len):
      output, hidden, cell, _ = self.decoder(x, encoder_states, hidden, cell)

      outputs[t] = output
      # output shape = (N, answer_vocab_size)

      best_guess = output.argmax(1)

      x = target[t] if random.random() < teacher_force_ratio else best_guess

    return outputs