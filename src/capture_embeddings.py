import os, gin, torch
import numpy as np
from src.utils.model import LM
from torch.utils.tensorboard import SummaryWriter
from src.utils.dataset import Cord19
from datetime import datetime

@gin.configurable()
def capture_embeddings_and_state(run_name, dim, train_file='./.data/countries_train.txt', writer=None):
    # open training token file
    topic_tokens = []
    with open(train_file, 'r') as f:
        for tok in f.readlines():
            topic_tokens.append(tok.lower().strip('\n'))

    # train_iter, _, _ = torchtext.datasets.WikiText2.iters(batch_size=1, bptt_len=1)
    train_iter, val_iter, test_iter = Cord19.iters(batch_size=1, bptt_len=1)
    vocab = train_iter.dataset.fields['text'].vocab.itos

    model_file = None
    for file in os.listdir('models'):
        if run_name in file and file.split('.')[-1] == 'pth':
            model_file = './models/{}'.format(file)

    device = torch.device('cuda')

    model = LM(vocab_size=len(vocab), embedding_dim=dim, hidden_dim=dim, bidirectional_lstm=True).to(device)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    token_embeddings = []
    token_labels = []

    all_embeddings = []
    all_labels = []

    token_hidden_state = []
    token_hidden_state_labels = []

    count = 0
    print('start: {}'.format(datetime.now()))
    for batch in train_iter:
        seq = batch.text.to(device)
        label_idx = batch.target.flatten()[-1]
        label = vocab[label_idx.item()]

        _ = model(seq)

        if label.lower() in topic_tokens:
            hidden_state = model.hidden_state.detach().cpu().flatten().cpu().numpy()
            token_hidden_state.append(hidden_state)
            token_hidden_state_labels.append(label)

            if label not in token_labels:
                token_embeddings.append(model.embedding.detach().cpu().numpy())
                token_labels.append(label)

            if label not in all_labels:
                all_embeddings.append(model.embedding.detach().cpu().numpy())
                all_labels.append(label)
                count += 1
                if count % 1000 == 0:
                    print('{} embeddings added'.format(count))

        count += 1
        if count % 10000 == 0:
            print('{}: processed {}'.format(datetime.now(), count))

    token_embeddings = np.vstack(token_embeddings)
    all_embeddings = np.vstack(all_embeddings)
    token_hidden_state = np.vstack(token_hidden_state)

    torch.save((token_embeddings, token_labels), 'models/{}_token_embeddings.pt'.format(run_name))
    torch.save((all_embeddings, all_labels), 'models/{}_all_embeddings.pt'.format(run_name))
    torch.save((token_hidden_state, token_hidden_state_labels), 'models/{}_token_hidden_state.pt'.format(run_name))

    if writer is not None:
        writer.add_embedding(token_embeddings.squeeze(1), token_labels, tag='token_embeddings')
        writer.add_embedding(all_embeddings.squeeze(1), all_labels, tag='all_embeddings')
        writer.add_embedding(token_hidden_state, token_hidden_state_labels, tag='token_hidden_states')

if __name__ == '__main__':
    capture_embeddings_and_state(
        'cord19-100_26_05_2020-19_39_57',
        dim=100,
        writer=SummaryWriter('runs/cord19-100_26_05_2020-19_39_57'),
        train_file='src/risk_factors.txt'
    )
