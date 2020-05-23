import torch, gin
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable()
class LM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_lstm_layers=3, bidirectional_lstm=True, dropout=0.5):
        super(LM, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.drop = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_state = None
        self.embedding = None

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=bidirectional_lstm,
            dropout=dropout
        )

        if bidirectional_lstm:
            self.linear = nn.Linear(hidden_dim*2, vocab_size)
        else:
            self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        embedding = self.embeddings(seq)
        self.embedding = embedding

        X = self.drop(embedding)
        X, lstm_hidden_state = self.lstm(X)
        self.hidden_state = lstm_hidden_state[0][-1]

        X = self.drop(X)
        X = self.linear(X)
        X = X.view(-1, self.vocab_size)
        X = F.log_softmax(X, dim=1)
        return X