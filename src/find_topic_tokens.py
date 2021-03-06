import os
import gin
import numpy as np
import torch
from src.utils.model import LM
from scipy.spatial.distance import euclidean
import csv
from src.utils.dataset import Cord19
from datetime import datetime

@gin.configurable()
def find_topic_tokens(run_name, dim, hold_out_file='./.data/countries_test.txt', sigma=0.8):

    token_hidden_states, token_hidden_states_labels = torch.load('./models/{}_token_hidden_state.pt'.format(run_name))
    token_hidden_states_labels_lower = [l.lower() for l in token_hidden_states_labels]
    uniq_labels = set(token_hidden_states_labels_lower)

    hiddens_state_means = {}
    for lbl in uniq_labels:
        indicies = np.where(np.array(token_hidden_states_labels_lower) == lbl)
        mean = np.mean(token_hidden_states[indicies], axis=0).reshape(1, -1)
        hiddens_state_means[lbl] = mean

    mean_hidden_state = np.mean(token_hidden_states, axis=0).reshape(1, -1)

    hold_out = []
    with open(hold_out_file, 'r') as f:
        for tok in f.readlines():
            hold_out.append(tok.lower().strip('\n'))

    # load model
    model_file = None
    for file in os.listdir('models'):
        if run_name in file and '.pth' in file:
            model_file = './models/{}'.format(file)

    train_iter, val_iter, test_iter = Cord19.iters(batch_size=1, bptt_len=1)
    vocab = train_iter.dataset.fields['text'].vocab.itos

    device = torch.device('cuda' if torch.cuda else 'cpu')

    model = LM(embedding_dim=dim, hidden_dim=dim, vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)
    model.eval()

    dist_f = open('./results/{}_similarity.csv'.format(run_name), 'w')
    fieldnames = ['iter', 'token', 'all_similarity', 'is_hold_out']
    fieldnames = fieldnames + [lbl for lbl in hiddens_state_means.keys()]
    dist_writer = csv.DictWriter(dist_f, fieldnames=fieldnames)
    dist_writer.writeheader()

    # print('iter         token                   similarity      hold_out?')
    # print('------------+-----------------------+---------------+---------')
    count = 0
    print('start: {}'.format(datetime.now()))

    for batch in train_iter:
        seq = batch.text.to(device)
        label_idx = batch.target.flatten()[-1]
        label = vocab[label_idx.item()]

        _ = model(seq)

        hidden_state = model.hidden_state.detach().cpu().numpy()

        dist_all = euclidean(hidden_state, mean_hidden_state)
        dists = [euclidean(hidden_state, m) for m in hiddens_state_means.values()]

        sim_all = np.exp((-(dist_all ** 2.)) / sigma)
        sims = [np.exp((-(dist ** 2.)) / sigma) for dist in dists]

        iter = str(count)
        token = label
        is_hold_out = True if label.lower() in hold_out else False

        # print('{:6s}       {:15s}         {:.5f}         {:1s}'.format(iter, token, sim_all, '*' if is_hold_out else ''))
        row = {'iter': iter, 'token': token, 'all_similarity': sim_all, 'is_hold_out': is_hold_out}
        for i, lbl in enumerate(hiddens_state_means.keys()):
            row[lbl] = sims[i]

        dist_writer.writerow(row)

        count += 1
        if count % 10000 == 0:
            print('{}: processed: {}'.format(datetime.now(), count))

    dist_f.close()


if __name__ == '__main__':
    find_topic_tokens('cord19-100_26_05_2020-19_39_57', dim=100, hold_out_file='src/risk_factors_holdout.txt')