import torch


def train_epoch(device, dataloader, model, loss_fns, optimizer):
    model.train()
    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device).flatten(1)
        mu, sigma, pred = model(X)

        # Backprop and optimize
        kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))  # TODO use torch loss
        loss = loss_fns['reconstruction'](pred, X) + kl_div

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_epoch(device, dataloader, model, loss_fns):
    model.eval()
    loss = 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device).flatten(1)
            mu, sigma, pred = model(X)
            loss += loss_fns['reconstruction'](pred, X).item()  # TODO add kl
    loss /= len(dataloader)
    return loss
