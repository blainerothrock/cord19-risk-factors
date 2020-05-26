import os, gin
from datetime import datetime
import numpy as np
import torch, torchtext
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import create_supervised_evaluator, Events
from ignite.handlers import ModelCheckpoint
from ignite_utils import create_engine, create_evaluator
from ignite.metrics import Loss
from model import LM
from capture_embeddings import capture_embeddings_and_state

@gin.configurable()
def train(id, max_epochs, batch_size, learning_rate, momentum, context_window, model_path):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_iter, val_iter, test_iter = torchtext.datasets.WikiText2.iters(batch_size=batch_size, bptt_len=context_window)

    vocab = train_iter.dataset.fields['text'].vocab
    vocab_size = len(train_iter.dataset.fields['text'].vocab)

    model = LM(vocab_size=vocab_size)
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    trainer = create_engine(model, optimizer, loss_fn, device, vocab)
    evaluator = create_evaluator(model, metrics={'nll': Loss(loss_fn)}, device=device)

    id = '{}_{}'.format(id, datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
    writer = SummaryWriter('runs/{}'.format(id))

    checkpoint_handler = ModelCheckpoint(model_path, id, n_saved=1, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_loss(engine):
        writer.add_scalar('Iter/train_loss', engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_loss(engine):
        loss = engine.state.output
        epoch = engine.state.epoch
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Perplexity/train', np.exp(loss), epoch)
        print('----------------')
        print('Epoch %i:' % epoch)
        print('  - train loss: %.2f' % loss)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_val_loss(engine):
        epoch = engine.state.epoch
        evaluator.run(val_iter)
        metrics = evaluator.state.metrics
        loss = metrics['nll']
        perplexity = np.exp(loss)
        writer.add_scalar('Loss/val', loss, epoch)
        writer.add_scalar('Perplexity/val', perplexity, epoch)
        print('  - val loss:   %.2f' % loss)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_loss(engine):
        epoch = engine.state.epoch
        evaluator.run(test_iter)
        metrics = evaluator.state.metrics
        loss = metrics['nll']
        perplexity = np.exp(loss)
        writer.add_scalar('Loss/test', loss, epoch)
        writer.add_scalar('Perplexity/test', perplexity, epoch)
        print('  - test loss:   %.2f' % loss)

    @trainer.on(Events.COMPLETED)
    def capture(engine):
        capture_embeddings_and_state(id, checkpoint_handler.last_checkpoint, vocab.itos, writer, context_window, device=device)

    trainer.run(train_iter, max_epochs=max_epochs)


if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    train()
