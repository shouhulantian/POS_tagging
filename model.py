from torch.utils.data import Dataset, DataLoader
import torch
from dataset import PosDataset
import torch.nn as nn
import  torch.nn.functional as F

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, char_dim = 0):
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        input = self.embedding(sentence)
        rnn_output, _ = self.rnn(input.permute(1,0,2))
        output = self.linear(rnn_output)
        scores = F.log_softmax(output.squeeze(dim = 1), dim =1)
        return scores.squeeze()

# tagger = LSTMTagger(20,50,5,6)
# input = torch.randint(5,(15,))
# output = tagger(input)
# print(output)