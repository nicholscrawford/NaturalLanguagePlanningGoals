import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np


from PointCloudRenderer.point_to_rgb_mlp import MLPPointToRGBModule
from PointCloudRenderer.point_to_rgb_transformer import \
    TransformerPointsToRGBModule

from PointCloudRenderer.rgbd_dataloader import get_dataloader, get_dataset_and_cache, get_train_val_dl

def train(args):
    data_dir = get_dataset_and_cache()
    train_loader, val_loader = get_train_val_dl(data_dir, k_points=args.k, batch_size=args.batch_size)

    # create the lightning module
    model = TransformerPointsToRGBModule(k=args.k, nhead=args.num_heads,
                             num_layers=args.num_layers, dropout=args.dropout)

    # create the trainer
    trainer = pl.Trainer(max_epochs=args.max_epochs)

    # train the model
    trainer.fit(model, train_loader, val_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10, help='Number of input points')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads in the transformer')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the transformer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation data loaders')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs to train')

    args = parser.parse_args()

    train(args)
