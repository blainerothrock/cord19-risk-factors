import csv, os
import gin
import numpy as np
import torchtext
import torch
from model import LM
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine, euclidean
import csv

@gin.configurable()
def find_topic_tokens(run_name, dim, hold_out_file='./.data/countries_test.txt', sigma=0.5):

    token_embeddings, token_embeddings_labels = torch.load('./models/{}_token_embeddings.pt'.format(run_name))
    token_hidden_states, token_hidden_states_labels = torch.load('./models/{}_token_hidden_state.pt'.format(run_name))

    mean_hidden_state = np.mean(token_hidden_states, axis=0).reshape(1, -1)

    # nbrs = NearestNeighbors(n_neighbors=4, ).fit(np.array(token_embeddings))

    hold_out = []
    with open(hold_out_file, 'r') as f:
        for tok in f.readlines():
            hold_out.append(tok.lower().strip('\n'))

    #load model
    model_file = None
    for file in os.listdir('./models'):
        if run_name in file and '.pth' in file:
            model_file = './models/{}'.format(file)

    train_iter, _, test_iter = torchtext.datasets.WikiText2.iters(batch_size=1, bptt_len=1)
    vocab = test_iter.dataset.fields['text'].vocab.itos

    device = torch.device('cuda' if torch.cuda else 'cpu')

    model = LM(embedding_dim=dim, hidden_dim=dim, vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)
    model.eval()

    count = 0
    sim_labels = {}

    dist_f = open('./results/{}_similarity.csv'.format(run_name), 'w')
    fieldnames = ['iter', 'token', 'similarity', 'is_hold_out']
    dist_writer = csv.DictWriter(dist_f, fieldnames=fieldnames)
    dist_writer.writeheader()

    print('iter         token                   similarity      hold_out?')
    print('------------+-----------------------+---------------+---------')
    for batch in test_iter:
        count += 1
        seq = batch.text.to(device)
        label_idx = batch.target.flatten()[-1]
        label = vocab[label_idx.item()]

        _ = model(seq)

        hidden_state = model.hidden_state.detach().cpu().numpy()

        dist = euclidean(hidden_state, mean_hidden_state)
        sim = np.exp( ( -( dist ** 2. ) ) / sigma )

        if label not in token_hidden_states_labels and sim > 0.9:
            if label not in sim_labels:
                #     ''iter       token                    similarity     hold_out?''
                iter = str(count)
                token = label
                is_hold_out = True if label.lower() in hold_out else False
                print('{:6s}       {:15s}         {:.5f}         {:1s}'.format(iter, token, sim, '*' if is_hold_out else ''))
                dist_writer.writerow({'iter': iter, 'token': token, 'similarity': sim, 'is_hold_out': is_hold_out})
                sim_labels[label] = 1
            else:
                sim_labels[label] += 1

    dist_f.close()

    count_f = open('./results/{}_count.csv'.format(run_name), 'w')
    fieldnames = ['token', 'count', 'is_hold_out']
    count_writer = csv.DictWriter(count_f, fieldnames=fieldnames)
    count_writer.writeheader()

    print('token           count     hold_out?')
    print('---------------+---------+---------')
    for (label, count) in sorted(sim_labels.items(), key=lambda x: x[1], reverse=True):
        if count > 3:
            is_hold_out = True if label.lower() in hold_out else False
            print('{:15s} {:5s} {:1s}'.format(label, str(count), '*' if is_hold_out else ''))
            count_writer.writerow({'token': label, 'count': count, 'is_hold_out': is_hold_out})


        # if label not in token_embeddings_labels:
        #     distances = []
        #     embedding = model.embeddings(label_idx.to(device))
        #     for token_embedding in token_hidden_states:
        #         # distances.append(1 - cosine(hidden_state, token_embedding))
        #         dist = euclidean(hidden_state, token_embedding)
        #         distances.append(np.exp( ( -( dist ** 2. ) ) / sigma ))
        #
        #     # distance = 1 - cosine(hidden_state, mean_hidden_state)
        #
        #     max_dist = max(distances)
        #
        #     if max_dist > 0.999:
        #         if max_dist == 1.0:
        #             print()
        #         if label not in sim_labels:
        #             print(count, 'hidden_state:', label, token_hidden_states_labels[np.argmax(distances)], max_dist)
        #             # print('hidden_state:', count, label, distance)
        #             sim_labels.append(label)

            # if label.lower() in hold_out:
            #     print('\n****')
            #     print(count, label, token_hidden_states_labels[np.argmax(distances)], max_dist)
            #     # print(label, distance)
            #     print('\n****')

    print()


if __name__ == '__main__':
    find_topic_tokens('wikitext2-bidirectional-50_25_05_2020-15_17_28', dim=50, hold_out_file='./.data/punc-test.txt')