import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np


from PointCloudRenderer.point_to_rgb_mlp import MLPPointToRGBModule
from PointCloudRenderer.point_to_rgb_transformer import \
    TransformerPointToRGBModule

def train(args):
    # create train and validation datasets
    train_dataset = DummyDataset(args.num_train_samples)
    val_dataset = DummyDataset(args.num_val_samples)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # create the lightning module
    model = PointToRGBModule(k=args.k, num_heads=args.num_heads, hidden_size=args.hidden_size,
                             num_layers=args.num_layers, dropout=args.dropout)

    # create the trainer
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs)

    # train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10, help='Number of input points')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in the transformer')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden dimension in the transformer')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the transformer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation data loaders')
    parser.add_argument('--num_train_samples', type=int, default=1000, help='Number of samples in the training dataset')
    parser.add_argument('--num_val_samples', type=int, default=100, help='Number of samples in the validation dataset')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of epochs to train')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')

    args = parser.parse_args()

    train(args)
