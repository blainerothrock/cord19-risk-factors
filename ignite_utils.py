from ignite.engine import Engine
import torch


def create_engine(model, optimizer, loss_fn, device, vocab):
    model.to(device)

    def _update(engine, batch):
        model.train()
        model.zero_grad()

        seq = batch.text.to(device)
        target = batch.target.flatten().to(device)

        output = model(seq)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        return loss.item()

    return Engine(_update)


def create_evaluator(model, metrics=None, device=None):
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            seq = batch.text.to(device)
            target = batch.target.view(-1).to(device)
            pred = model(seq)

            return pred, target

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
