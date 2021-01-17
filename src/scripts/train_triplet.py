import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, random_split

from src.models import TripletNet
from src.utils import parse_args, visualize_embeddings
from src.config import ModelType, MODEL_DICT, LOSS_DICT

def get_model(model_args, train_args, input_shape, device):
    model_args = vars(model_args)
    model_args['loss_type'] = LOSS_DICT[model_args['loss_type']]
    if MODEL_DICT[train_args.model] == ModelType.Triplet:
        model = TripletNet(input_shape=input_shape,
                           lr=train_args.lr, 
                           device=device,
                           **model_args)
    return model

if __name__ == "__main__":

    model_args, train_args, data_args = parse_args()

    # data loaders
    train_dataset = MNIST(os.getcwd(), 
                          train=True, 
                          download=True, 
                          transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True)
    test_dataset = MNIST(os.getcwd(), 
                          train=False, 
                          download=True, 
                          transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]))
    test_loader = DataLoader(test_dataset, batch_size=train_args.batch_size, shuffle=False)

    # Initialize a trainer
    trainer = pl.Trainer(gpus=1, max_epochs=train_args.epochs, progress_bar_refresh_rate=20)
    device =  torch.device(f"cuda:{trainer.root_gpu}") if trainer.on_gpu and trainer.root_gpu is not None else torch.device('cpu')

    # Init Model
    # get input shape
    x_train, _ = next(iter(train_loader))
    input_shape = x_train.shape[1:]
    model = get_model(model_args, train_args, input_shape, device)

    # visualize the embeddings before training
    visualize_embeddings(model, test_loader, device, title="{}_{}_{}".format(0, model_args.loss_type, train_args.model))

    # Train the model ⚡
    trainer.fit(model, train_loader)

    # visualize the embeddings post training
    visualize_embeddings(model, 
                         test_loader, 
                         device, 
                         title="{}_{}_{}".format(train_args.epochs, model_args.loss_type, train_args.model))
