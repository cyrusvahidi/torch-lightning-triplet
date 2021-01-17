from argparse import ArgumentParser

def parse_args():
    model_args = ArgumentParser()
    train_args = ArgumentParser()
    data_args = ArgumentParser()

    # data-related arguments
    data_args.add_argument(
        "--audio_path", type=str, help="path to dataset", default="/import/c4dm-datasets/FSD50Knumpy/npy"
    )

    # model hyperparams
    model_args.add_argument(
        "--embedding_size", type=int, help="number of embedding dimensions", default=4
    )
    model_args.add_argument(
        "--loss_type", 
        type=str, 
        help="loss to use", 
        default='online_batch_all',
        choices=['triplet', 'online_batch_all', 'online_batch_hard'],
    )
    model_args.add_argument(
        "--margin", 
        type=float, 
        help="triplet loss margin", 
        default=0.4,
    )
    model_args.add_argument(
        "--squared", 
        type=bool, 
        help="use squared euclidean distances for embeddings", 
        default=True,
    )
    model_args.add_argument(
        "--normalize_embeddings", 
        type=bool, 
        help="l2-normalize embeddings before computing triplet loss", 
        default=False,
    )

    # Training hyperparameter arguments
    train_args.add_argument(
        "--model", 
        type=str, 
        help="model type to train", 
        default='triplet',
        choices=['mnist', 'triplet'],
    )
    train_args.add_argument(
        "--lr", type=float, help="learning rate", default=1e-4
    )
    train_args.add_argument(
        "--epochs", type=int, help="maximum epochs", default=100
    )
    train_args.add_argument(
        "--batch_size", type=int, help="batch size", default=256
    )


    return model_args.parse_args(), train_args.parse_args(), data_args.parse_args() 