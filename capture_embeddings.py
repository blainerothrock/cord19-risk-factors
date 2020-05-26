import os, gin, torch
import torchtext
import numpy as np
from model import LM
from sklearn.neighbors import NearestNeighbors
from torch.utils.tensorboard import SummaryWriter


@gin.configurable()
def capture_embeddings_and_state(run_name, dim, train_file='./.data/countries_train.txt', writer=None):
    # open training token file
    topic_tokens = []
    with open(train_file, 'r') as f:
        for tok in f.readlines():
            topic_tokens.append(tok.lower().strip('\n'))

    train_iter, _, _ = torchtext.datasets.WikiText2.iters(batch_size=1, bptt_len=1)
    vocab = train_iter.dataset.fields['text'].vocab.itos

    device = torch.device('cuda' if torch.cuda else 'cpu')

    model_file = None
    for file in os.listdir('./models'):
        if run_name in file:
            model_file = './models/{}'.format(file)

    model = LM(vocab_size=len(vocab), embedding_dim=dim, hidden_dim=dim, bidirectional_lstm=True).to(device)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)
    model.eval()

    token_embeddings = []
    token_labels = []

    all_embeddings = []
    all_labels = []

    token_hidden_state = []
    token_hidden_state_labels = []

    count = 0
    for batch in train_iter:
        seq = batch.text.to(device)
        label_idx = batch.target.flatten()[-1]
        label = vocab[label_idx.item()]

        _ = model(seq)

        if label.lower() in topic_tokens:
            if label not in token_hidden_state_labels:
                hidden_state = model.hidden_state.detach().flatten().cpu().numpy()
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

    token_embeddings = np.vstack(token_embeddings)
    all_embeddings = np.vstack(all_embeddings)
    token_hidden_state = np.vstack(token_hidden_state)

    torch.save((token_embeddings, token_labels), 'models/{}_token_embeddings.pt'.format(run_name))
    torch.save((all_embeddings, all_labels), 'models/{}_all_embeddings.pt'.format(run_name))
    torch.save((token_hidden_state, token_hidden_state_labels), 'models/{}_token_hidden_state.pt'.format(run_name))

    if writer is not None:
        writer.add_embedding(token_embeddings, token_labels, tag='token_embeddings')
        writer.add_embedding(all_embeddings, all_labels, tag='all_embeddings')
        writer.add_embedding(token_hidden_state, token_hidden_state_labels, tag='token_hidden_states')

if __name__ == '__main__':
    capture_embeddings_and_state(
        'wikitext2-bidirectional-50_25_05_2020-15_17_28',
        dim=50,
        writer=SummaryWriter('./runs/wikitext2-bidirectional-50_25_05_2020-15_17_28'),
        train_file='./.data/punc-train.txt')
