import os, gin, torch
import torchtext
import numpy as np
from model import LM


@gin.configurable(whitelist=['train_file'])
def capture_embeddings_and_state(identifier, model_file, vocab, writer, context_window=1, train_file='',
                                 device=torch.device('cpu')):
    # open training token file
    topic_tokens = []
    with open(train_file, 'r') as f:
        for tok in f.readlines():
            topic_tokens.append(tok.lower().strip('\n'))

    train_iter, _, _ = torchtext.datasets.WikiText2.iters(batch_size=1, bptt_len=context_window)

    model = LM(vocab_size=len(vocab)).to(device)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)
    model.eval()

    token_embeddings = []
    token_labels = []

    all_embeddings = []
    all_labels = []

    token_hidden_state = []
    token_hidden_state_labels = []

    for batch in train_iter:
        seq = batch.text.to(device)
        label_idx = batch.target.flatten()[-1]
        label = vocab[label_idx.item()]
        embedding = model.embeddings(label_idx.to(device)).detach().cpu().numpy()

        _ = model(seq)

        if label.lower() in topic_tokens:
            hidden_state = model.hidden_state.detach().flatten().cpu().numpy()
            token_hidden_state.append(hidden_state)
            token_hidden_state_labels.append(label)

            if label not in token_labels:
                token_embeddings.append(embedding)
                token_labels.append(label)

        if label not in all_labels:
            all_embeddings.append(embedding)
            all_labels.append(label)

    # torch.save(token_embeddings, 'models/{}_token_embeddings.pt')
    # torch.save(all_embeddings, 'models/{}_all_embeddings.pt')
    # torch.save(token_hidden_state, 'models/{}_token_hidden_state.pt')

    token_embeddings = np.vstack(token_embeddings)
    all_embeddings = np.vstack(all_embeddings)
    token_hidden_state = np.vstack(token_hidden_state)

    if writer is not None:
        writer.add_embedding(token_embeddings, token_labels, tag='token_embeddings')
        writer.add_embedding(all_embeddings, all_labels, tag='all_embeddings')
        writer.add_embedding(token_hidden_state, token_hidden_state_labels, tag='token_hidden_states')
