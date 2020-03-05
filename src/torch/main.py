"""Script to train the Attention model."""
import argparse

import os
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

sys.path.append(os.getcwd())
from src.datasets.data import ComposeTransformations, PositionalEncoding, \
    to_observation_tuples, Physionet2019Dataset
from src.torch.models.attention_model import AttentionModel
from src.torch import models
from src.torch.torch_utils import variable_length_collate


def train(args, model, device, train_loader, optimizer, epoch):
    """Train model."""
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, labels = batch['ts'], batch['labels']
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Flatten outputs to support nll_loss
        output = output.reshape(-1, 2)
        labels = labels.reshape(-1)

        label_weight = \
            labels[:, np.newaxis] == torch.tensor([0, 1], device=device)[np.newaxis, :]

        label_weight = label_weight.sum(dim=0, dtype=torch.float)
        label_weight = label_weight.sum() / label_weight

        loss = F.nll_loss(output, labels, weight=label_weight)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    n_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            data = batch['ts']
            labels = batch['labels']
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            # Flatten labels and model output
            output = output.reshape(-1, 2)
            labels = labels.reshape(-1)
            # Sum up batch loss
            test_loss += F.nll_loss(output, labels, reduction='sum').item()
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            n_predictions += (labels != -100).sum()

    test_loss /= float(n_predictions)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, n_predictions,
        100. * correct / n_predictions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AttentionModel', metavar='M',
                        help='Model class to be used (default: AttentionModel)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--hypersearch', action='store_true', default=False,
                        help='perform hypersearch, generate parameters from random grid')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device: {device}')

    transform = ComposeTransformations([
        PositionalEncoding(1, 250000, 20),
        to_observation_tuples
    ])
    full_dataset = Physionet2019Dataset(transform=transform)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(
        full_dataset, [train_size, test_size]
    )

    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=variable_length_collate,
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=variable_length_collate,
        batch_size=args.test_batch_size, shuffle=True, **kwargs
    )

    model_cls = args.model
    #TODO: instanciate model_class with classmethod decorator to set hyperparameters
    params = {  'd_in': train_dataset[0]['ts'].shape[-1], 
                'hypersearch': args.hypersearch}
    
    model = getattr(models, model_cls).set_hyperparams(params).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
