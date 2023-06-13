import torch


def compute_loss(device, x, model, construction_loss_fn):
    # Forward
    x = x.to(device).flatten(1)
    mu, sigma, pred = model(x)

    # Compute loss
    kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
    loss = construction_loss_fn(pred, x) + kl_div
    return loss


def train_epoch(device, dataloader, model, construction_loss_fn, optimizer):
    model.train()
    for batch, (x, _) in enumerate(dataloader):
        loss = compute_loss(device, x, model, construction_loss_fn)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_epoch(device, dataloader, model, construction_loss_fn):
    model.eval()
    loss = 0
    with torch.no_grad():
        for x, _ in dataloader:
            loss += compute_loss(device, x, model, construction_loss_fn)
    loss /= len(dataloader.dataset)
    return loss
