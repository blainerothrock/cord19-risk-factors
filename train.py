import os, gin
import numpy as np
import torch
import torchtext
from model import LM

@gin.configurable()
def train(max_epochs, batch_size, learning_rate, context_window):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_iter, val_iter, test_iter = torchtext.datasets.WikiText2.iters(batch_size=batch_size, bptt_len=context_window)

    vocab = train_iter.dataset.fields['text'].vocab
    vocab_size = len(train_iter.dataset.fields['text'].vocab)

    model = LM(vocab_size=vocab_size)
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model = model.to(device)
    model.train()

    for epoch in range(max_epochs):
        running_loss = []
        iter_count = 0
        for item in train_iter:
            seq = item.text.to(device)
            target = item.target.view(-1).to(device)

            model.zero_grad()

            output = model(seq)

            loss = loss_fn(output, target)
            running_loss.append(loss.item())
            iter_count += 1
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print('Epoch: %i\n   - Loss: %.2f' % (epoch, np.mean(running_loss)))
            running_loss = []
            iter_count = 0

if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    train()
