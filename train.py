import os, gin
import numpy as np
import torch, torchtext
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import create_supervised_evaluator, Events
from ignite_utils import create_engine, create_evaluator
from ignite.metrics import Loss
from model import LM

@gin.configurable()
def train(max_epochs, batch_size, learning_rate, momentum, context_window):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_iter, val_iter, test_iter = torchtext.datasets.WikiText2.iters(batch_size=batch_size, bptt_len=context_window)

    vocab = train_iter.dataset.fields['text'].vocab
    vocab_size = len(train_iter.dataset.fields['text'].vocab)

    model = LM(vocab_size=vocab_size)
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    trainer = create_engine(model, optimizer, loss_fn, device)
    evaluator = create_evaluator(model, metrics={'nll': Loss(loss_fn)}, device=device)

    writer = SummaryWriter()

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
        print('  - test loss:  %.2f' % loss)

    trainer.run(train_iter, max_epochs=max_epochs)


if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    train()
