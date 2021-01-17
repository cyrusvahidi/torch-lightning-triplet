import os

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from src.losses import online_batch_all

from src.config import LossType

from .architectures import CNNEmbedding

class TripletNet(pl.LightningModule):

    def __init__(self, 
                 input_shape=(), 
                 embedding_size=4, 
                 lr=1e-3, 
                 device=None,
                 loss_type=LossType.ONLINE_BATCH_ALL, 
                 margin=0.5,
                 squared=True,
                 normalize_embeddings=False):
        super(TripletNet, self).__init__()

        self.save_hyperparameters()

        self.model = CNNEmbedding(input_shape, embedding_size)

        self.lr = lr

        self.device_id = device

        # triplet loss parameters
        self.loss_type = loss_type
        self.margin = margin
        self.squared = squared
        self.normalize_embeddings = normalize_embeddings

    def _compute_loss(self, embeddings, labels):
        if self.loss_type == LossType.ONLINE_BATCH_ALL:
            loss_metrics = online_batch_all(embeddings, 
                                            labels, 
                                            margin=self.margin, 
                                            squared=self.squared, 
                                            normalize=self.normalize_embeddings,
                                            device=self.device_id)
            loss = loss_metrics[0]
            self.log('num_positive_triplets', loss_metrics[1], prog_bar=True)
            self.log('num_valid_triplets', loss_metrics[2], prog_bar=True)
            return loss

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, labels = batch
        # x = x.view(x.size(0), -1)
        embeddings = self(x)
        loss = self._compute_loss(embeddings, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        x = x.view(x.size(0), -1)
        embeddings = self(x)
        loss = self.loss(embeddings, labels)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer